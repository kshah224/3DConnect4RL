import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from connect4env import *
from functools import reduce
import wandb
from tqdm import tqdm

def discount_rewards(rewards, gamma=0.85):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

def rollout(model, env, max_steps=1000, temperature=1.0, p2=False):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs, info = env.reset()
    valid_actions = info["valid_actions"]
    
    ep_reward = 0
    for _ in range(max_steps):
      logits, val = model(torch.tensor([obs.flatten()], dtype=torch.float32, device=DEVICE))
      
      if temperature != 1.0:
        logits /= temperature

      # Mask logits based on valid actions
      action_mask = torch.from_numpy(valid_actions.flatten()).to(DEVICE)

      # Apply mask
      logits = torch.where(action_mask, logits, -float('inf'))
      
      act_distribution = Categorical(logits=logits)
      act = act_distribution.sample()
      act_log_prob = act_distribution.log_prob(act).item()

      act, val = act.item(), val.item()

      next_obs, reward, done, info, = env.step(act)

      if p2:
        reward *= -1

      valid_actions = info["valid_actions"]

      for i, item in enumerate((obs, act, reward, val, act_log_prob)):
        train_data[i].append(item)

      obs = next_obs
      ep_reward += reward

      if done:
        break

    train_data = [np.asarray(x) for x in train_data]

    ### Do train data filtering
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, ep_reward

# Policy and value model
class ActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size, action_space_size):
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
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

class PPOTrainer():
  def __init__(self, actor_critic, ppo_clip_val=0.2, target_kl_div=0.01, max_policy_train_iters=80, value_train_iters=80, policy_lr=3e-4, value_lr=1e-2):
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
    self.policy_optim = optim.AdamW(policy_params, lr=policy_lr)

    value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layers.parameters())
    self.value_optim = optim.AdamW(value_params, lr=value_lr)

  def train_policy(self, obs, acts, old_log_probs, gaes, _wandb=False):
    total_loss = 0
    for _ in range(self.max_policy_train_iters):
      self.policy_optim.zero_grad()

      new_logits = self.ac.policy(obs)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(acts)

      policy_ratio = torch.exp(new_log_probs - old_log_probs)
      clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
      
      clipped_loss = clipped_ratio * gaes
      full_loss = policy_ratio * gaes
      policy_loss = -torch.min(full_loss, clipped_loss).mean()
      
      total_loss += policy_loss.item()

      policy_loss.backward()
      self.policy_optim.step()

      kl_div = (old_log_probs - new_log_probs).mean()
      if kl_div >= self.target_kl_div:
        break
      
    if _wandb:
        wandb.log({"policy_loss": total_loss})

  def train_value(self, obs, returns, _wandb=False):
    total_loss = 0
    for _ in range(self.value_train_iters):
      self.value_optim.zero_grad()

      values = self.ac.value(obs)
      value_loss = (returns - values) ** 2
      value_loss = value_loss.mean()
      
      total_loss += value_loss.item()

      value_loss.backward()
      self.value_optim.step()
  
    if _wandb:
      wandb.log({"value_loss": total_loss})
      
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WANDB = True
    
    env = Connect4Env()
    model = ActorCriticNetwork(reduce(lambda x,y: x*y, env.observation_space.shape), env.action_space.n)
    model = model.to(DEVICE)
    
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    n_episodes = int(1e6)
    print_freq = 200
    save_freq = int(5e3)

    ppo = PPOTrainer(model, policy_lr = 3e-4, value_lr = 1e-3, target_kl_div = 0.02, max_policy_train_iters = 80, value_train_iters = 80)
    
    # Training loop
    ep_rewards = []
    max_temp = 5.0
    min_temp = 1.0
    end_freq = 1e4
    
    if WANDB:
        run = wandb.init(project="connect4-rl", name=f"p1_max_temp{max_temp}_{end_freq}_episodes{n_episodes}_model_size{model_size}")
    
    for episode_idx in tqdm(range(n_episodes)):
        # Perform rollout
        temperature = max(min_temp, max_temp - (max_temp - min_temp) * episode_idx / end_freq)

        train_data, reward = rollout(model, env, temperature=temperature, p2=False)
        ep_rewards.append(reward)

        # Shuffle
        permute_idxs = np.random.permutation(len(train_data[0]))

        # Policy data
        obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
        obs = obs.flatten(1)
        acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
        act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

        # Value data
        returns = discount_rewards(train_data[2])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # Train model
        if (episode_idx + 1) % print_freq == 0:
          ppo.train_policy(obs, acts, act_log_probs, gaes, _wandb=WANDB)
          ppo.train_value(obs, returns, _wandb=WANDB)
        else:
          ppo.train_policy(obs, acts, act_log_probs, gaes)
          ppo.train_value(obs, returns)
          
        if (episode_idx + 1) % print_freq == 0:
            # print('Episode {} | Avg Reward {:.1f}'.format(
            #     episode_idx + 1, np.mean(ep_rewards[-print_freq:])))
            
            if WANDB:
                wandb.log({"avg_reward": np.mean(ep_rewards[-print_freq:]), 
                           "temperature" : temperature
                           })
        
        if (episode_idx + 1) % save_freq == 0:
          torch.save(model.state_dict(), f"./weights/p1_model_ep{episode_idx}_reward{np.mean(ep_rewards):.1f}_action_mask_F.pt")
            
    