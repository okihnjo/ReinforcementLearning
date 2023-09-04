import numpy as np

from agent import Agent
import plotly.graph_objects as go
import numpy as np
import laserhockey.hockey_env as hock_env
from utils_sac import load_model
import torch.nn as nn
import pandas as pd

# run this python file to see my best two agents play against the weak opponent. The result will be saved in the eval_folder

def play(eps: int, agent1: nn.Linear):
    env = hock_env.HockeyEnv()
    global games_played

    load_model(agent1.actor_local, "05.08.2023/21.48.57/sac_hockey.pt") # best model against weak, used in tournament
   # load_model(agent1.actor_local, "09.08.2023/21.30.28/sac_hockey.pt") # second best against weak, also quite strong

    opponent = hock_env.BasicOpponent(weak=True)
    agent1.actor_local.eval()     
    stats = []
    for episode in range(eps):
        state,_ = env.reset()
        obs_opponent = env.obs_agent_two()
        while True:
            #env.render()
            action_ag_1 = agent1.act(state)
            opponent_action = opponent.act(obs_opponent)
            next_state, reward, done,_, info = env.step(np.hstack([action_ag_1,opponent_action]))
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
                    mode='lines',
                    name='WeakOpponent'))
    fig.update_layout(title=f"Win percentage over {number_games} games. Endscore: {end_score_meins}:{end_score_opponent}, draws: {draws}",)
    fig.show()
    go.Figure.write_image(fig, f"eval/images/myStrongVSopponentWeakdfsfs.png")


env_for_shape = hock_env.HockeyEnv()
state_size = env_for_shape.observation_space.shape[0]
agent = Agent(state_size=state_size, action_size=4,hidden_size=256,random_seed=0) 

if __name__ == "__main__":
    end_score_meins = 0
    end_score_opponent = 0
    draws = 0
    games_played = 0
    percentage_wins_mine, percentage_wins_opponent = [], []
    for s in range(100):
        for i in range(4):
            games_played+=1
            play(1, agent)
        percentage_wins_opponent.append(end_score_opponent/games_played)
        percentage_wins_mine.append(end_score_meins/games_played)
    my_pd = pd.DataFrame({"my_agent":percentage_wins_mine, "opponent":percentage_wins_opponent})
    plot_win_percentage(my_pd, games_played)

    print(f"end_score_meins: {end_score_meins}, end_score_opponent: {end_score_opponent}, draws: {draws}")
