import gymnasium as gym
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import time
import os
import gc
import ale_py

gym.register_envs(ale_py)

SAVE_EVERY = 5
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def stack_frames(stacked_frames: deque, frame: np.ndarray, is_new_episode: bool, stack_size: int = 4):
    frame = preprocess_frame(frame)
    if is_new_episode or stacked_frames is None:
        stacked_frames = deque([np.zeros((84,84), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=2), stacked_frames

class AttentionLayer(layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features: tf.Tensor, hidden: tf.Tensor):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W(features) + self.U(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights

class DuelingDARQN(tf.keras.Model):
    def __init__(self, input_shape, action_size, lstm_units=64, attention_units=32):
        super().__init__()
        self.conv1 = layers.Conv2D(16, 8, strides=4, activation='relu')
        self.conv2 = layers.Conv2D(32, 4, strides=2, activation='relu')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(128, activation='relu')
        self.lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.attention = AttentionLayer(attention_units)
        self.value = layers.Dense(1)
        self.advantage = layers.Dense(action_size)

    def call(self, x, hidden_state=None, cell_state=None):
        batch_size, seq_len, h, w, c = x.shape
        x = tf.reshape(x, (-1, h, w, c))
        x = self.conv1(x)
        x = self.conv2(x)
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

class ReplayBuffer:
    def __init__(self, size, seq_len, input_shape):
        self.size = size
        self.ptr = 0
        self.full = False
        self.seq_len = seq_len
        self.states = np.zeros((size, seq_len, *input_shape), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, seq_len, *input_shape), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(self, s_seq, a, r, next_s_seq, d):
        self.states[self.ptr] = s_seq
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = next_s_seq
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_i = self.size if self.full else self.ptr
        idx = np.random.choice(max_i, batch_size, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])

class Agent:
    def __init__(self, input_shape, seq_len, action_size, gamma=0.99, lr=0.0005):
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
        self.loss_fn = tf.keras.losses.Huber()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, seq_state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        q_values, _, _, _ = self.model(np.expand_dims(seq_state, axis=0))
        return int(tf.argmax(q_values[0]).numpy())

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        next_q_main, _, _, _ = self.model(next_states)
        next_actions = tf.argmax(next_q_main, axis=1)
        next_q_target, _, _, _ = self.target_model(next_states)
        indices = tf.stack([tf.range(tf.shape(next_q_target)[0]), tf.cast(next_actions, tf.int32)], axis=1)
        max_next_q = tf.gather_nd(next_q_target, indices)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            q, _, _, _ = self.model(states)
            indices = tf.stack([tf.range(tf.shape(q)[0]), actions], axis=1)
            q_action = tf.gather_nd(q, indices)
            loss = self.loss_fn(target_q, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = tf.convert_to_tensor(states, tf.float32)
        next_states = tf.convert_to_tensor(next_states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)
        return self.train_step(states, actions, rewards, next_states, dones)

STACK_SIZE, SEQ_LEN = 4, 4
INPUT_SHAPE = (84,84,STACK_SIZE)
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.99995
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 1000
WARMUP_STEPS = 1000
MAX_STEPS = 30000
TRAIN_EVERY = 4
LOG_INTERVAL = 200
EPISODE_LOG_INTERVAL = 1

env = gym.make("ALE/MsPacman-v5", render_mode=None)
ACTION_SIZE = env.action_space.n

agent = Agent(INPUT_SHAPE, SEQ_LEN, ACTION_SIZE)
buffer = ReplayBuffer(10000, SEQ_LEN, INPUT_SHAPE)

state, _ = env.reset()
stacked_frames = None
state, stacked_frames = stack_frames(stacked_frames, state, True)
seq_buffer = deque([state]*SEQ_LEN, maxlen=SEQ_LEN)

episode_reward = 0
episode = 0
epsilon = EPS_START
rewards_history = []
losses = []

start_time_total = time.time()
print("Entraînement DARQN démarré !")

for step in range(MAX_STEPS):
    seq_state = np.stack(seq_buffer, axis=0)
    action = agent.act(seq_state, epsilon)

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    next_state_stack, stacked_frames = stack_frames(stacked_frames, next_state, False, STACK_SIZE)
    seq_buffer.append(next_state_stack)

    buffer.add(seq_state, action, reward, np.stack(seq_buffer, axis=0), float(done))
    episode_reward += reward

    if step > WARMUP_STEPS and buffer.ptr > BATCH_SIZE and step % TRAIN_EVERY == 0:
        batch = buffer.sample(BATCH_SIZE)
        loss = agent.train(batch)
        losses.append(loss.numpy())

    if step % LOG_INTERVAL == 0 and losses:
        avg_loss = np.mean(losses)
        elapsed_total = time.time() - start_time_total
        print(f"Step {step} | Avg Loss: {avg_loss:.4f} | Elapsed: {int(elapsed_total)}s | Buffer: {buffer.ptr}/{buffer.size}")
        losses = []

    if done:
        episode += 1
        rewards_history.append(episode_reward)
        print(f"Episode {episode:3d} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f} | Buffer: {buffer.ptr}/{buffer.size}")

        if episode % SAVE_EVERY == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"darqn_episode_{episode}.weights.h5")
            agent.model.save_weights(save_path)
            print(f"💾 Modèle sauvegardé : {save_path}")

        state, _ = env.reset()
        stacked_frames = None
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        seq_buffer = deque([state]*SEQ_LEN, maxlen=SEQ_LEN)
        episode_reward = 0
        gc.collect()
        tf.keras.backend.clear_session()

    if step > WARMUP_STEPS:
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

final_model_path = os.path.join(CHECKPOINT_DIR, f"darqn_final.weights.h5")
agent.model.save_weights(final_model_path)
print(f"💾 Modèle final sauvegardé : {final_model_path}")

env.close()
print("Entraînement terminé !")
