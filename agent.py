import numpy as np
from train_a2c import ActorCriticNet
from train_ppo import ActorCriticNetwork 
from dqn_network import DQN
from connect4env import Connect4Env
from functools import reduce
import torch
from minimax import alphabeta

class Agent:
    def __init__(self) -> None:
        pass

    def get_action(self, env):
        raise NotImplementedError
    
    def save(self, filename):
        pass

    def load(self, filename):
        pass

class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
    
    def get_action(self, env):
        moves = env.get_legal_moves()
        return np.random.choice(moves,1)[0]

class MinimaxAgent(Agent):
    def __init__(self, depth, player) -> None:
        super().__init__()
        self.depth = depth
        self.player = player

    def get_action(self, env, step=-1):
        if step == 0:
            moves = env.get_legal_moves()
            return np.random.choice(moves,1)[0]
        return alphabeta(self.player, env, self.depth)[0]

class PPOAgent(Agent):
    def __init__(self, chkp_path, env, device) -> None:
        super().__init__()
        self.model = ActorCriticNetwork(reduce(lambda x,y: x*y, env.observation_space.shape), env.action_space.n).to(device)
        self.model.load_state_dict(torch.load(chkp_path, map_location=device))
        self.model.eval()
        self.device = device

    def get_action(self, env):
        obs = env.board
        valid_actions = env.get_valid_action_mask()
        logits, value = self.model(torch.tensor(np.array([obs.flatten()]), dtype=torch.float32, device=self.device))

        action_mask = torch.from_numpy(valid_actions.flatten()).to(self.device)
        action_masked = torch.where(action_mask, logits, -float('inf'))
        action = torch.softmax(action_masked.squeeze(0), dim=0)
        action_taken = torch.argmax(action).item()

        return action_taken
    
class A2CAgent(Agent):
    def __init__(self, chkp_path, env, device) -> None:
        super().__init__()
        self.model = torch.load(chkp_path, map_location=device)
        self.model.eval()
        self.device = device

    def get_action(self, env):
        obs = env.board
        valid_actions = env.get_valid_action_mask()
        logits, value = self.model(torch.tensor(np.array([obs.flatten()]), dtype=torch.float32, device=self.device))

        action_mask = torch.from_numpy(valid_actions.flatten()).to(self.device)
        action_masked = torch.where(action_mask, logits, -float('inf'))
        action = torch.softmax(action_masked.squeeze(0), dim=0)
        action_taken = torch.argmax(action).item()

        return action_taken
    
class DQNAgent(Agent):
    def __init__(self, chkp_path, env, device) -> None:
        super().__init__()
        self.model = torch.load(chkp_path, map_location=device)
        self.model.eval()
        self.device = device

    def get_action(self, env):
        obs = env.board
        valid_actions = env.get_valid_action_mask()
        logits = self.model(torch.tensor(np.array([obs.flatten()]), dtype=torch.float32, device=self.device))

        action_mask = torch.from_numpy(valid_actions.flatten()).to(self.device)
        action_masked = torch.where(action_mask, logits, -float('inf'))
        action = torch.softmax(action_masked.squeeze(0), dim=0)
        action_taken = torch.argmax(action).item()

        return action_taken  
