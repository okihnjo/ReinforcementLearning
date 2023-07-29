import numpy as np

import gym
from collections import deque
import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import plotly.express as px
from agent import Agent
from simple_agent import SimpleAgent
from utils_sac import plot_reward
import plotly.graph_objects as go
import numpy as np
import gymnasium as gym
from importlib import reload
import copy
import datetime
import laserhockey.hockey_env as hock_env
from utils_sac import  moving_mean, save_network, load_model, plot_reward

np.set_printoptions(suppress=True)
writer = SummaryWriter("runs/"+'test')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def SAC(n_episodes=500, max_t=500, print_every=50, agent_type="new"):
    actor_losses, critic_losses, critic2_losses, alpha_losses = [], [], [], []
    opponent = hock_env.BasicOpponent(weak=False)
    scores = []
    touched = 0
    agent_1 = hock_env.BasicOpponent(weak=True)
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        obs_agent2 = env.obs_agent_two()
        score = 0
        first_time_touch = 1
        s = -0.5
        for t in range(max_t):
            action_ag_1 = agent_1.act(state)
            action_ag_2 = opponent.act(obs_agent2)
            # action_ag haben shape 4.0
            # input shape ist aber 8.0 laut game engine
            env.render()
            next_state, reward, done,_, info = env.step(np.hstack([action_ag_1,action_ag_2]))
            if s<info['reward_closeness_to_puck']:
                s = info['reward_closeness_to_puck']
                reward= reward + 5
            else: 
                reward = reward - 1
            
            losses=agent.step(state, action_ag_1, reward, next_state, done, t)
            if losses != None:
                actor_losses.append(losses[0])
                critic_losses.append(losses[1])
                critic2_losses.append(losses[2])
                alpha_losses.append(losses[3])
            state = next_state
            obs_agent2 = env.obs_agent_two()
            score += reward

            if done:
                break 
        
        scores.append(score)
        writer.add_scalar("Reward", score, i_episode)
        writer.add_scalar("average_X", np.mean(scores), i_episode)
        if i_episode % print_every == 0:      
            plot_reward(scores)
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f} '.format(i_episode, score, np.mean(scores)), end="")
    moving_mean((actor_losses, critic_losses, critic2_losses, alpha_losses))
    if agent_type=="new" : save_network(agent.actor_local, "sac")





def play():
    agent.actor_local.eval()
    epis = []
    for i_episode in range(4):

        state = env.reset()

        state = state.reshape((1,state_size))

        while True:
            env.render()
            action = agent.act(state)
            action_v = action.numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state.reshape((1,state_size))
            state = next_state
            epis.append(reward)
            if done:
                print(len(epis))
                break 
                


parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="hock_env", help="Environment name")
parser.add_argument("-ep", type=int, default=500, help="The amount of training episodes, default is 100")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=100, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--agent_type", type=str, default="new", help="If new, then double q is used. Use given old, then simple version is executed")
parser.add_argument("--compare", type=bool, default=False, help="If true, compare the two agents")

args = parser.parse_args()
reload(hock_env)

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed
    n_episodes = args.ep
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size        # minibatch size
    LR_ACTOR = args.lr         # learning rate of the actor 
    LR_CRITIC = args.lr        # learning rate of the critic
    FIXED_ALPHA = args.alpha
    saved_model = args.saved_model
    agent_type = args.agent_type
    comp_flag = args.compare

    t0 = time.time()
    env = hock_env.HockeyEnv()
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    torch.manual_seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=4, random_seed=seed,hidden_size=HIDDEN_SIZE, action_prior="uniform") #"normal"
    SAC(n_episodes=args.ep, max_t=250, print_every=args.print_every, agent_type=args.agent_type)
    t1=time.time()
    env.close()
    print("training took {} min!".format((t1-t0)/60))
