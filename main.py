import gymnasium as gym
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import ale_py
import time
import gc
import os
import glob

gym.register_envs(ale_py)

# ------------------------------- CONFIG
save_every = 10          # Sauvegarde du modèle tous les N épisodes
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# ------------------------------- Prétraitement des frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def stack_frames(stacked_frames, frame, is_new_episode, stack_size=4):
    frame = preprocess_frame(frame)
    if is_new_episode or stacked_frames is None:
        stacked_frames = deque([np.zeros((84,84), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=2), stacked_frames

# ------------------------------- Attention Layer
class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W(features) + self.U(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights

# ------------------------------- Dueling DARQN avec CNN + LSTM + Attention
class DuelingDARQN(tf.keras.Model):
    def __init__(self, input_shape, action_size, lstm_units=128, attention_units=64):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(256, activation='relu')
        self.lstm = layers.LSTM(lstm_units, return_state=True, return_sequences=True)
        self.attention = AttentionLayer(attention_units)
        self.value = layers.Dense(1)
        self.advantage = layers.Dense(action_size)

    def call(self, x, hidden_state=None, cell_state=None):
        batch_size, seq_len, h, w, c = x.shape
        x = tf.reshape(x, (-1, h, w, c))
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = tf.reshape(x, (batch_size, seq_len, -1))
        if hidden_state is None or cell_state is None:
            lstm_out, h, c = self.lstm(x)
        else:
            lstm_out, h, c = self.lstm(x, initial_state=[hidden_state, cell_state])
        context_vector, attn_weights = self.attention(lstm_out, h)
        v = self.value(context_vector)
        a = self.advantage(context_vector)
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q, h, c, attn_weights

# ------------------------------- Replay Buffer léger
class ReplayBuffer:
    def __init__(self, size, seq_len, input_shape):
        self.size = size
        self.ptr = 0
        self.full = False
        self.seq_len = seq_len
        self.input_shape = input_shape
        self.states = np.zeros((size, seq_len, *input_shape), dtype=np.uint8)
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, seq_len, *input_shape), dtype=np.uint8)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(self, s_seq, a, r, next_s_seq, d):
        self.states[self.ptr] = (s_seq*255).astype(np.uint8)
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = (next_s_seq*255).astype(np.uint8)
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_i = self.size if self.full else self.ptr
        idx = np.random.choice(max_i, batch_size, replace=False)
        states = self.states[idx].astype(np.float32)/255.0
        next_states = self.next_states[idx].astype(np.float32)/255.0
        return (states, self.actions[idx], self.rewards[idx], next_states, self.dones[idx])

# ------------------------------- Agent DARQN + Double DQN
class Agent:
    def __init__(self, input_shape, seq_len, action_size, gamma=0.99, lr=0.00025):
        self.seq_len = seq_len
        self.action_size = action_size
        self.gamma = gamma
        self.model = DuelingDARQN(input_shape, action_size)
        self.target_model = DuelingDARQN(input_shape, action_size)
        dummy = tf.zeros((1, seq_len, *input_shape), dtype=tf.float32)
        _ = self.model(dummy)
        _ = self.target_model(dummy)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.update_target()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, seq_state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        q_values, _, _, _ = self.model(np.expand_dims(seq_state, axis=0))
        return int(tf.argmax(q_values[0]).numpy())

    def train(self, batch, target_update_freq, step):
        states, actions, rewards, next_states, dones = batch
        states = tf.convert_to_tensor(states, tf.float32)
        next_states = tf.convert_to_tensor(next_states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)

        next_q_main, _, _, _ = self.model(next_states)
        next_actions = tf.argmax(next_q_main, axis=1)
        next_q_target, _, _, _ = self.target_model(next_states)
        indices = tf.stack([tf.range(tf.shape(next_q_target)[0]), tf.cast(next_actions, tf.int32)], axis=1)
        max_next_q = tf.gather_nd(next_q_target, indices)
        target_q = rewards + (1-dones)*self.gamma*max_next_q

        with tf.GradientTape() as tape:
            q, _, _, _ = self.model(states)
            indices = tf.stack([tf.range(tf.shape(q)[0]), actions], axis=1)
            q_action = tf.gather_nd(q, indices)
            loss = tf.reduce_mean(tf.square(target_q - q_action))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        if step % target_update_freq == 0:
            self.update_target()
        return loss.numpy()

# ------------------------------- Hyperparamètres
stack_size, seq_len = 4, 4
input_shape = (84,84,stack_size)
eps_min = 0.01
eps_decay = 0.998
batch_size = 32
target_update_freq = 500
warmup_steps = 500
max_steps = 10000
train_every = 2

# ------------------------------- Environnement
train_env = gym.make("ALE/MsPacman-v5", render_mode=None)
action_size = train_env.action_space.n

agent = Agent(input_shape, seq_len, action_size)
buffer = ReplayBuffer(3000, seq_len, input_shape)

# ------------------------------- Charger le dernier modèle si disponible
weight_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")))
if weight_files:
    latest_model = weight_files[-1]
    agent.model.load_weights(latest_model)
    agent.update_target()
    # Valeur d'epsilon lors de la reprise (ne pas repartir à 1.0)
    epsilon = 0.05  
    print(f"🔄 Reprise de l'entraînement depuis le modèle : {latest_model} | epsilon={epsilon}")
else:
    epsilon = 1.0

# ------------------------------- Initialisation
state, info = train_env.reset()
stacked_frames = None
state, stacked_frames = stack_frames(stacked_frames, state, True)
seq_buffer = deque([state]*seq_len, maxlen=seq_len)

episode_reward = 0
episode = 0
rewards_history = []
losses = []
log_interval = 200
episode_log_interval = 2
start_time = time.time()

print("Entraînement DARQN démarré !")

# ------------------------------- Boucle principale
for step in range(max_steps):
    seq_state = np.stack(seq_buffer, axis=0)
    action = agent.act(seq_state, epsilon)

    next_state, reward, terminated, truncated, info = train_env.step(action)
    done = terminated or truncated
    next_state_stack, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
    seq_buffer.append(next_state_stack)

    buffer.add(seq_state, action, reward, np.stack(seq_buffer, axis=0), float(done))
    episode_reward += reward

    # Entraînement
    if step > warmup_steps and buffer.ptr > batch_size and step % train_every == 0:
        try:
            batch = buffer.sample(batch_size)
            loss = agent.train(batch, target_update_freq, step)
            losses.append(loss)
        except ValueError:
            pass

    # Logging
    if step % log_interval == 0 and losses:
        avg_loss = np.mean(losses)
        elapsed = time.time() - start_time
        print(f"Step {step} | Avg Loss: {avg_loss:.4f} | Elapsed: {elapsed/60:.2f} min | Epsilon: {epsilon:.3f} | Buffer: {buffer.ptr}/{buffer.size}")
        losses = []
        start_time = time.time()

    # Fin d'épisode
    if done:
        episode += 1
        rewards_history.append(episode_reward)

        if episode % episode_log_interval == 0:
            avg_reward = np.mean(rewards_history[-episode_log_interval:])
            best_reward = np.max(rewards_history)
            print(f"Episode {episode:3d} | Reward: {episode_reward:.1f} | Avg({episode_log_interval}): {avg_reward:.1f} | Best: {best_reward:.1f} | Epsilon: {epsilon:.3f} | Buffer: {buffer.ptr}/{buffer.size}")

        # Sauvegarde du modèle
        if episode % save_every == 0:
            save_path = os.path.join(checkpoint_dir, f"darqn_episode_{episode}.weights.h5")
            agent.model.save_weights(save_path)
            print(f"💾 Modèle sauvegardé : {save_path}")

        # Réinitialisation pour le nouvel épisode
        state, info = train_env.reset()
        stacked_frames = None
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        seq_buffer = deque([state]*seq_len, maxlen=seq_len)
        episode_reward = 0
        if episode % 10 == 0:
            gc.collect()

    # Décroissance epsilon
    if step > warmup_steps:
        epsilon = max(eps_min, epsilon * eps_decay)

train_env.close()
print("Entraînement terminé !")
