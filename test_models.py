from train_ppo import ActorCriticNetwork
from dqn_network import DQN
from connect4env import Connect4Env
from agent import RandomAgent, PPOAgent, MinimaxAgent, A2CAgent, DQNAgent
from functools import reduce
import torch
import numpy as np
from tqdm import tqdm

def main():
    PATH_PPO = "weights/model_ep234999_reward-0.3_action_mask.pt"
    PATH_A2C = "weights/a2c_net5.pth"
    PATH_DQN = "weights/target_net.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Connect4Env()
    ppo_agent = PPOAgent(PATH_PPO, env, DEVICE)
    a2c_agent = A2CAgent(PATH_A2C, env, DEVICE)
    dqn_agent = DQNAgent(PATH_DQN, env, DEVICE)
    minimax_agent = MinimaxAgent(depth=2, player=1)
    random_agent = RandomAgent()
    VERBOSE = False

    print_freq = 1
    wins = 0
    loss = 0
    n_test = 10
    for episode_idx in tqdm(range(n_test)):
        ep_reward = 0
        obs, info = env.reset()

        for step in range(1000):
            # Change here for different model
            action_taken = a2c_agent.get_action(env)
            next_obs, reward, done, info = env.step(action_taken)

            obs = next_obs
            ep_reward += reward

            if done:
                if env.check_win(verbose=VERBOSE):
                    wins += 1
                else:
                    loss += 1
                break

            # Change here for different model
            action = dqn_agent.get_action(env)
            next_obs, reward, done, info, = env.step(action)
            
            ep_reward += reward
            obs = next_obs

            if done:
                if env.check_win(verbose=VERBOSE):
                    loss += 1
                else:
                    wins += 1
                break
        
        if VERBOSE:
            print(obs)

        if (episode_idx + 1) % print_freq == 0:
            print('Episode {} | Steps {} | Reward {:.1f} | Wins P1 {} | Wins P2 {}'.format(episode_idx + 1, step+1, ep_reward, wins, loss))


    print("WINS P1: {}".format(wins))
    print("WINS P2: {}".format(loss))

if __name__ == "__main__":
    main()
