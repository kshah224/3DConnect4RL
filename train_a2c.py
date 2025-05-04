import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym
from connect4env import *
from functools import reduce
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Policy and value model
class ActorCriticNet(nn.Module):
  def __init__(self, obs_space_size, action_space_size):
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU())
    
    self.policy_layers = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_space_size))
    
    self.value_layers = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1))
    
  def value(self, obs):
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value
        
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value
  
# Define the Actor-Critic Trainer
class ActorCriticTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update(self, obs, acts, returns, values, advantages):
        self.optimizer.zero_grad()
        policy_logits = self.model.policy(obs.to(device))
    
        policy_dist = Categorical(logits=policy_logits)
        log_probs = policy_dist.log_prob(acts)
        policy_loss = -(log_probs * advantages).mean()
        
        value_loss = nn.MSELoss()(returns, values).mean()
        # value_loss = nn.SmoothL1Loss()(returns, value)
        # entropy_loss = (policy_dist.entropy()).mean()
        loss = policy_loss + value_loss 
        
        loss.backward()
        self.optimizer.step()

        return loss.item()
    def save_model(self):
        torch.save(self.model, "weights/a2c_net5.pth")
        

    def load_model(self):
        self.model = torch.load("weights/a2c_net3.pth").to(self.device)
        

def calculate_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    advantages = []
    advantage = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] - values[t]
        advantage = delta + gamma * lambda_ * advantage
        advantages.append(advantage)
    advantages.reverse()
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

if __name__ == "__main__":
        
    # Initialize the environment and model
    env = Connect4Env()
    model = ActorCriticNet(reduce(lambda x,y: x*y, env.observation_space.shape), env.action_space.n)
    model.to(device)
    trainer = ActorCriticTrainer(model)



    # Training Loop
    num_episodes = 250000
    gamma = 0.99
    print_freq = 100
    rewards = []
    loss = []
    writer = SummaryWriter()
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        episode_values = []
        # model.zero_grad()
        while not done:
            obs_tensor = torch.tensor([obs.flatten()], dtype=torch.float32)
            obs_tensor = obs_tensor.to(device)
            policy_logits, value = model(obs_tensor)
            action = Categorical(logits=policy_logits).sample().item()

            next_obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            episode_values.append(value.item())

            obs = next_obs
        rewards.append(sum(episode_rewards))
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        for r, v in zip(reversed(episode_rewards), reversed(episode_values)):
            R = r + gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - v)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8).to(device)
        
        # advantages = torch.tensor(advantages, dtype=torch.float32)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = calculate_advantages(episode_rewards, episode_values).to(device)
        values = torch.tensor(episode_values, dtype=torch.float32).to(device)
        # values = (values - values.mean()) / (values.std() + 1e-8)

        obs_tensor = torch.tensor([obs.flatten()], dtype=torch.float32).to(device)
        ep_loss = trainer.update(obs_tensor, torch.tensor([action]).to(device), returns, values, advantages)
        loss.append(ep_loss)
        # Print episode info
        if (episode + 1) % print_freq == 0:
            trainer.save_model()
            writer.add_scalar('Average Reward', np.mean(rewards[-print_freq:]), episode+1)
            writer.add_scalar('Average Loss', np.mean(loss[-print_freq:]), episode+1)
        
            print(f"Episode: {episode+1}, Avg Reward: {np.mean(rewards[-print_freq:])}, Loss: {np.mean(loss[-print_freq:])}")

    writer.close()