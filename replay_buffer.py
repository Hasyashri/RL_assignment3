"""
replay_buffer.py
----------------
Stores past experiences (state, action, reward, next_state, done)
so that the agent can learn from random mini-batches.
"""

import numpy as np
from collections import deque
from typing import Tuple

class ReplayBuffer:
    """
    Simple experience replay buffer for DQN.
    Stores experiences in a first-in-first-out manner.
    """
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add one experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Randomly sample a batch of experiences."""
        assert len(self.buffer) >= batch_size, "Not enough samples in buffer!"
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))

        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)      # Shape: (B, 4, H, W)
        next_states = np.array(next_states, dtype=np.float32)

        # Remove any extra channel dimension (H, W, 1 -> H, W)
        if states.ndim == 5 and states.shape[-1] == 1:
            states = np.squeeze(states, axis=-1)
            next_states = np.squeeze(next_states, axis=-1)

        return (
            states,
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            next_states,
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        """Return current number of stored experiences."""
        return len(self.buffer)
