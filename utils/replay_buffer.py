"""Fixed-size experience replay buffer backed by pre-allocated numpy arrays."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = int(capacity)
        self.state_size = int(state_size)

        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Store one transition, overwriting the oldest one when the buffer is full."""
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = 1.0 if done else 0.0

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Return a random batch as tensors on the requested device."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        states = torch.from_numpy(self.states[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).unsqueeze(1).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        next_states = torch.from_numpy(self.next_states[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size