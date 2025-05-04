import torch
import torch.nn.functional as F
from torch import nn

class DQN(nn.Module):
    def __init__(self, n_observations = 168, n_actions= 42, hidden_layer = 128):
        """
        Initialize a deep Q-learning network 
        Arguments:
            n_observations: number of observations.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, n_actions)

    def forward(self, obs):
        
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x