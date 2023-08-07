import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils_sac import hidden_init
import numpy as np
import random
from buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim
from networks import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = 0.95
TAU = 1e-2
HIDDEN_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256        # minibatch size
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4       # learning rate of the critic
FIXED_ALPHA = None

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, hidden_size, action_prior="uniform"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed) # maybe I have to do another seed for q2 ??
        self.seed_2 = random.seed(random_seed+1)

        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 0.2
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR) 
        self._action_prior = action_prior
        
        print("Using: ", device)
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed+1, hidden_size).to(device)
        
        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed+1,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=0) 

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        

    def step(self, state, action, reward, next_state, done, step) -> None or tuple:
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            return self.learn(step, experiences, GAMMA)
        else:
            return None
    
    def act(self, obs):
        """Returns actions for given state as per current policy."""
        if str(type(obs)) == "<class 'numpy.ndarray'>":
            state = torch.from_numpy(obs).float().to(device)
        else:
            state = torch.FloatTensor(obs).to(device).unsqueeze(0)
        action = self.actor_local.get_action(state).detach()
        return action.numpy() # von action zu action[0] und [0] macht kein sinn

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state)) Seite 8
        Critic_loss = MSE(Q, Q_target) Seite 9
        Actor_loss = α * log_pi(a|s) - Q(s,a) Seite 10
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, log_pis_next = self.actor_local.evaluate(next_states)

            Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
            Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

            # take the mean of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)

            if FIXED_ALPHA == None:
                # Compute Q targets for current states (y_i) Seite 15 - The inclusion of (1 - dones) allows SAC to properly handle terminal 
                # states by excluding the discount factor for terminal states and considering only the immediate rewards in those cases.
                Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
            else:
                Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            actions_pred, log_pis = self.actor_local.evaluate(states)
            q_1_pi = self.critic1(states, actions_pred)
            q_2_pi = self.critic2(states, actions_pred)
            min_q_pi = torch.min(q_1_pi, q_2_pi)
            actor_loss = (self.alpha * log_pis - min_q_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if FIXED_ALPHA == None:
                # Compute alpha loss
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean() # Seite 16
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
                # Compute actor loss
                #actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:
                actor_loss = (FIXED_ALPHA * log_pis - min_q_pi).mean()
                #actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, TAU)
            self.soft_update(self.critic2, self.critic2_target, TAU)
            return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), alpha_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)