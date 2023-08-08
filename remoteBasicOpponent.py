import numpy as np

from laserhockey.hockey_env import BasicOpponent
from RL2023HockeyTournamentClient.client.remoteControllerInterface import RemoteControllerInterface
from RL2023HockeyTournamentClient.client.backend.client import Client
import laserhockey.hockey_env as hock_env
from utils_sac import load_model

from agent import Agent
class RemoteBasicOpponent(BasicOpponent, RemoteControllerInterface):

    env_for_shape = hock_env.HockeyEnv()
    state_size = env_for_shape.observation_space.shape[0]
    action_size = env_for_shape.action_space.shape[0]
    my_agent = Agent(state_size=state_size, action_size=4, random_seed=0,hidden_size=256, action_prior="uniform") #"normal"
    load_model(my_agent.actor_local, "07.08.2023/14.31.37/sac_hockey.pt")

    def __init__(self, weak, keep_mode=True):
        self.agent = RemoteBasicOpponent.my_agent
        BasicOpponent.__init__(self, weak=weak, keep_mode=keep_mode)
        RemoteControllerInterface.__init__(self, identifier='Okihnjo')

    def remote_act(self,
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.agent.act(obs)


if __name__ == '__main__':
    controller = RemoteBasicOpponent(weak=False)
    # Play n (None for an infinite amount) games and quit
    client = Client(username='LosNinos',
                    password='chahT7nei8',
                    controller=controller,
                    output_path='logs/basic_opponents', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
