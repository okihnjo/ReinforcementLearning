import numpy as np
import torch
import time
import argparse
import plotly.express as px
from agent import Agent
from utils_sac import plot_reward
import plotly.graph_objects as go
import numpy as np
from importlib import reload
import laserhockey.hockey_env as hock_env
from utils_sac import  moving_mean, save_network, load_model, plot_reward
import random
import torch.nn as nn
import pandas as pd
def play_2(eps: int, agent1: nn.Linear, agent2: nn.Linear):
    env = hock_env.HockeyEnv()
    global games_played
    load_model(agent1.actor_local, "08.08.2023/11.25.38/sac_hockey.pt")
    load_model(agent2.actor_local, "07.08.2023/14.31.37/sac_hockey.pt")
    agent1.actor_local.eval()     
    agent2.actor_local.eval()
    stats = []
    for i_episode in range(eps):
        state,_ = env.reset()
        obs_opponent = env.obs_agent_two()
        while True:
            #env.render()
            action_ag_1 = agent1.act(state)
            action_ag_2 = agent2.act(obs_opponent)
            next_state, reward, done,_, info = env.step(np.hstack([action_ag_1,action_ag_2]))
            state = next_state
            obs_opponent = env.obs_agent_two()
            if done:
                stats.append(info['winner'])
                break 
    evaluate_game(stats, eps)      


def evaluate_game(stats: list, eps: int):
    global end_score_meins, end_score_opponent, draws
    #print(f"\nplayed {eps} games, agent won {stats.count(1)} times, opponent won {stats.count(-1)} times, draw {stats.count(0)} times\n")
    if stats.count(1) > stats.count(-1):
        end_score_meins += 1
    elif stats.count(1) < stats.count(-1):
        end_score_opponent += 1
    else:
        draws += 1

def plot_win_percentage(df: pd.DataFrame, number_games: int):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["my_agent"].index, y=df["my_agent"],
                    mode='lines',
                    name='myAgent'))
    fig.add_trace(go.Scatter(x=df["opponent"].index, y=df["opponent"],
                    mode='lines+markers',
                    name='opponent'))
    fig.show()
# Show the plot

env_for_shape = hock_env.HockeyEnv()
action_high = env_for_shape.action_space.high[0]
action_low = env_for_shape.action_space.low[0]
state_size = env_for_shape.observation_space.shape[0]
action_size = env_for_shape.action_space.shape[0]
agent = Agent(state_size=state_size, action_size=4,hidden_size=256,random_seed=0, action_prior="uniform") #"normal"
agent_2 = Agent(state_size=state_size, action_size=4,hidden_size=256, random_seed=0,action_prior="uniform") #"normal"

if __name__ == "__main__":
    end_score_meins = 1
    end_score_opponent = 1
    draws = 1
    games_played = 0
    percentage_wins_mine, percentage_wins_opponent = [], []
    for s in range(40):
        for i in range(10):
            games_played+=1
            play_2(1, agent, agent_2)
        percentage_wins_opponent.append(end_score_opponent/games_played)
        percentage_wins_mine.append(end_score_meins/games_played)
    my_pd = pd.DataFrame({"my_agent":percentage_wins_mine, "opponent":percentage_wins_opponent})
    plot_win_percentage(my_pd, games_played)
    print(f"end_score_meins: {end_score_meins}, end_score_opponent: {end_score_opponent}")
