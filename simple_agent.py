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
GAMMA = 0.99
TAU = 1e-2
HIDDEN_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256        # minibatch size
LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 5e-4       # learning rate of the critic
FIXED_ALPHA = None

class SimpleAgent():
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
        self.seed = random.seed(random_seed)
        
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR) 
        self._action_prior = action_prior
        
        print("Using: ", device)
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        
        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=0)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        

    def step(self, state, action, reward, next_state, done, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(step, experiences, GAMMA)
            
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(state).detach()
        return action

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
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        

        # take the mean of both critics for updating
        
        if FIXED_ALPHA == None:
            # Compute Q targets for current states (y_i) Seite 15 - The inclusion of (1 - dones) allows SAC to properly handle terminal 
            # states by excluding the discount factor for terminal states and considering only the immediate rewards in those cases.
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target1_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target1_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean() # Seite 16
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha
                # Compute actor loss
                actor_loss = (alpha * log_pis - Q_target1_next).mean()
                #actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:
                actor_loss = (FIXED_ALPHA * log_pis - Q_target1_next).mean()
                #actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, TAU)

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