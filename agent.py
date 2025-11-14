"""
agent.py
---------
Defines the DQN agent that chooses actions using an epsilon-greedy strategy.
"""

import torch
import numpy as np
import random

class DQNAgent:
    """
    DQN Agent that selects actions with epsilon-greedy exploration.
    """
    def __init__(self, n_actions: int, device: torch.device):
        self.n_actions = n_actions
        self.device = device

    def select_action(self, state: np.ndarray, model: torch.nn.Module, epsilon: float) -> int:
        """
        Chooses an action:
            - With probability ε → random action (explore)
            - Otherwise → greedy action (exploit)
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = model(state_t)
        return q_values.argmax(1).item()
