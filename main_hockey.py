import numpy as np
import torch
import time
import argparse
from agent import Agent
from utils_sac import plot_reward
import numpy as np
from importlib import reload
import laserhockey.hockey_env as hock_env
from utils_sac import  moving_mean, save_network, load_model, plot_reward
import random

np.set_printoptions(suppress=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor_losses, critic_losses, critic2_losses, alpha_losses, scores, overall_stats = [], [], [], [], [], []
def SAC(n_episodes=500, max_t=500, agent_type="new", mode: int=0):
    print("\n------------------------------\n")
    print(f"Playing {n_episodes} episodes with max_t={max_t} with mode {mode} \n")
    env=hock_env.HockeyEnv(mode=mode)
    str_oppponent_cnt = 0
    t_threshold = 8 # threshold for how soon we reward agent when it touches puck
    p=10000
    opponent = hock_env.BasicOpponent(weak=True)    
    agent_support = hock_env.BasicOpponent(weak=False)
    puck_not_touch_penalty = 0.2 # penalty for not touching puck
    closeness_intensity = 6 # the bigger this value the more we punish him from being far away from puck
    closeness_before = 0 # check if we are closer than before
    little_extra_reward = 0.05
    for episode in range(1, n_episodes+1):
        state, _ = env.reset()
        obs_agent2 = env.obs_agent_two()
        score = 0
        total_reward = 0
        first_touch_flag = False
        for t in range(max_t):
            #env.render()
            if random.random() < p: # bootstrap strong opponent
                p = p * 0.5
                action_ag_1 = agent_support.act(state) 
                str_oppponent_cnt += 1   
            else:
                action_ag_1 = agent.act(state)
            action_ag_2 = opponent.act(obs_agent2)
            next_state, reward, done,_, info = env.step(np.hstack([action_ag_1,action_ag_2]))
            if info['reward_touch_puck'] == 1 and first_touch_flag is False:
                reward += puck_not_touch_penalty * t # he touched the puck (finally), reward him by reversing the punishment
                first_touch_flag = True # but do it only once, set flag
                if t<t_threshold: # threshold
                    reward += little_extra_reward # reward him for touching puck early, not useful when he touches late in game
            if not first_touch_flag: # as long as he doesnt touch the puck, punish him
                reward -= puck_not_touch_penalty            
            reward+= (closeness_intensity * info['reward_closeness_to_puck']) # je näher desto weniger schlechter / the closer the less worse
            if info['reward_closeness_to_puck'] > closeness_before: # check if we are closer than before
                reward += little_extra_reward # let him stay aggressive, if he is closer than before, reward him a little bit
            closeness_before = info['reward_closeness_to_puck']
            total_reward += reward
            losses=agent.add_transition(state, action_ag_1, reward, next_state, done, t)
            if losses != None:
                actor_losses.append(losses[0])
                critic_losses.append(losses[1])
                critic2_losses.append(losses[2])
                alpha_losses.append(losses[3])
            state = next_state
            obs_agent2 = env.obs_agent_two()
            score += reward
            total_steps = t
            if done:
                break 
        if p<0.01:
            if episode<100:
                p=100
            else: 
                p=10
        print(f"\r Moves made by STRONG opponent {str_oppponent_cnt}, by total {total_steps} ({str(str_oppponent_cnt/total_steps)}%) ", end="") if total_steps != 0 else print(f"\r 0 moves made ", end="")
        str_oppponent_cnt = 0
        scores.append(score)
    env.close()
    if agent_type=="new" : save_network(agent.actor_local, "sac_hockey") 

def play_hockey(eps: int):
    env=hock_env.HockeyEnv()
    agent.actor_local.eval()
    opponent = hock_env.BasicOpponent(weak=True)
    epis = []
    stats = []
    for episode in range(eps):
        state,_ = env.reset()
        obs_opponent = env.obs_agent_two()
        while True:
            #env.render()
            action_ag_1 = agent.act(state)
            action_ag_2 = opponent.act(obs_opponent)
            next_state, reward, done,_, info = env.step(np.hstack([action_ag_1,action_ag_2]))
            state = next_state
            obs_opponent = env.obs_agent_two()
            epis.append(reward)
            if done:
                stats.append(info['winner'])
                break 
    env.close()
    evaluate_game(stats, eps)           

def evaluate_game(stats: list, eps: int):
    print(f"\nplayed {eps} games, agent won {stats.count(1)} times, opponent won {stats.count(-1)} times, draw {stats.count(0)} times\n")
    overall_stats.append(stats.count(1)/eps)

parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="100", help="Environment name")

parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate of adapting the network weights, default is 1e-4")
parser.add_argument("-a", "--alpha",  type=float, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor gamma, default is 0.99")
parser.add_argument("-saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--agent_type", type=str, default="new", help="If new, then double q is used. Use given old, then simple version is executed")

args = parser.parse_args()
reload(hock_env)

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed
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
    t0 = time.time()
    env_for_shape = hock_env.HockeyEnv()
    action_high = env_for_shape.action_space.high[0]
    action_low = env_for_shape.action_space.low[0]
    torch.manual_seed(seed)
    env_for_shape.seed(seed)
    np.random.seed(seed)
    state_size = env_for_shape.observation_space.shape[0]
    action_size = env_for_shape.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=4, random_seed=seed,hidden_size=HIDDEN_SIZE)
    env_for_shape.close()
    if saved_model != None:
        load_model(agent.actor_local, saved_model)
        for i in range(10):
            play_hockey(eps=10)
    else:
        SAC(n_episodes=1000, max_t=800, agent_type=args.agent_type, mode=2)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        SAC(n_episodes=1000, max_t=800, agent_type=args.agent_type, mode=0)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(30)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=1)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=2)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=2000, max_t=800, agent_type=args.agent_type, mode=0)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(30)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=1)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(30)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=2)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(30)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=1000, max_t=800, agent_type=args.agent_type, mode=0)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(40)
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=1)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(40)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=2)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(40)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=1000, max_t=800, agent_type=args.agent_type, mode=0)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=1)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=500, max_t=800, agent_type=args.agent_type, mode=2)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=2000, max_t=800, agent_type=args.agent_type, mode=0)
        SAC(n_episodes=750, max_t=800, agent_type=args.agent_type, mode=1)
        print("\nTraining finished. Playing now for evaluation:\n")
        play_hockey(20)
        print("\nPlaying finished. We start training again..\n")
        print("-----------------------\n")
        SAC(n_episodes=1000, max_t=800, agent_type=args.agent_type, mode=2)
        print("-----------------------\n")
        SAC(n_episodes=1000, max_t=800, agent_type=args.agent_type, mode=0)
        print("\nTraining finished. Playing now for evaluation:\n")
        moving_mean((actor_losses, critic_losses, critic2_losses, alpha_losses), window_size=400)
        for i in range(10):
            play_hockey(30)
    
        window_size = 40
        num_segments = len(scores) // window_size
        segments = np.array_split(scores, num_segments)
        averages = [segment.mean() for segment in segments]
        plot_reward(averages, "reward")
        plot_reward(overall_stats, "overall_game_stats")
    t1=time.time()
    print("\ntraining took {} min!".format((t1-t0)/60))
