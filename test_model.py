import gymnasium as gym
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import ale_py
import os

gym.register_envs(ale_py)

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

class Agent:
    def __init__(self, input_shape, seq_len, action_size):
        self.seq_len = seq_len
        self.action_size = action_size
        self.model = DuelingDARQN(input_shape, action_size)
        dummy = tf.zeros((1, seq_len, *input_shape), dtype=tf.float32)
        _ = self.model(dummy)

    def act(self, seq_state, epsilon=0.05):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        q_values, _, _, _ = self.model(np.expand_dims(seq_state, axis=0))
        return int(tf.argmax(q_values[0]).numpy())

STACK_SIZE, SEQ_LEN = 4, 4
INPUT_SHAPE = (84,84,STACK_SIZE)
NUM_EPISODES = 5
EPSILON = 0.05
CHECKPOINT_PATH = "checkpoints/darqn_final.weights.h5"

env = gym.make("ALE/MsPacman-v5", render_mode="human")
ACTION_SIZE = env.action_space.n

agent = Agent(INPUT_SHAPE, SEQ_LEN, ACTION_SIZE)
agent.model.load_weights(CHECKPOINT_PATH)
print(f"✅ Modèle chargé depuis : {CHECKPOINT_PATH}")

for ep in range(1, NUM_EPISODES+1):
    state, _ = env.reset(seed=np.random.randint(0,10000))
    stacked_frames = None
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    seq_buffer = deque([state + np.random.normal(0,0.01,state.shape) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    done = False
    total_reward = 0

    while not done:
        seq_state = np.stack(seq_buffer, axis=0)
        action = agent.act(seq_state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_stack, stacked_frames = stack_frames(stacked_frames, next_state, False, STACK_SIZE)
        seq_buffer.append(next_state_stack)
        total_reward += reward

    print(f"Episode {ep} | Score total : {total_reward}")

env.close()
