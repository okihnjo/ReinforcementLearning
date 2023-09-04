import torch
import torch.nn.functional as F
import numpy as np
import random
from buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
from networks import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = 0.95
TAU = 1e-2
HIDDEN_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256        # minibatch size
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_ALPHA = 1e-4
LR_CRITIC = 1e-4       # learning rate of the critic
FIXED_ALPHA = None

class Agent():
    
    def __init__(self, state_size, action_size, random_seed, hidden_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed) # maybe do another seed for q2
        self.seed_2 = random.seed(random_seed+1)

        self.target_entropy = -action_size
        self.alpha = 0.2
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ALPHA) 
        
        print("Using: ", device)

        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)     
        
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed+1, hidden_size).to(device)
        
        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed+1,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=0) 

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    def act(self, obs) -> np.ndarray:
        """Given state s, following our policy π, we get an action a ∼ π(.|s) and return it.

        Args:
            obs (ndarray or tensor): observation/state from our environment.

        Returns:
            ndarray: action
        """
        if str(type(obs)) == "<class 'numpy.ndarray'>":
            state = torch.from_numpy(obs).float().to(device)
        else:
            state = torch.FloatTensor(obs).to(device).unsqueeze(0)
        action = self.actor_local.get_action(state).detach()
        return action.numpy()
        

    def add_transition(self, state, action, reward, next_state, done, step) -> None or tuple:
        """Save experience and transition knowledge in buffer.

        Args:
            state: current state
            action: chosen action in that state
            reward: reward for that action in that state
            next_state: next state after taking that action
            done: if the episode is done or not
            step: current step in episode

        Returns:
            None or tuple: If enough samples are available in memory, return tuple of losses.
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            return self.learn(experiences, GAMMA)
        else:
            return None
    

    def learn(self, experiences, gamma):
        """If enough samples in buffer, update parameters (actor, critic and alpha). \n
        q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state)) (Paper equ. 5) \n
        critic_loss = MSE(Q, Q_target) (Paper equ. 8) \n
        policy_loss = α * log_pi(a|s) - Q(s,a) (Paper equ. 10)
        alpha_loss = - (α * (log_pi(a|s) + H)) (Paper equ. 11)
       
        Args:
            experiences (tuples): tuple of experiences tuples, see class Experience.
            gamma (float): discount factor

        Returns:
            Losses from actor, critics and temperatur alpha.
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_action, log_pis_next = self.actor_local.evaluate(next_states)

            q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
            q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

            # take the minimum of both critics for updating
            q_target_next = torch.min(q_target1_next, q_target2_next)

            if FIXED_ALPHA == None:
                q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
            else:
                q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (q_target_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
        # Compute critic loss
        q_1 = self.critic1(states, actions).cpu()
        q_2 = self.critic2(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(q_1, q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(q_2, q_targets.detach())
        # Update both critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # update actor and alpha if alpha is not fixed
        actions_pred, log_pis = self.actor_local.evaluate(states)
        q_1_pi = self.critic1(states, actions_pred)
        q_2_pi = self.critic2(states, actions_pred)
        min_q_pi = torch.min(q_1_pi, q_2_pi)   
        if FIXED_ALPHA == None:
            actor_loss = (self.alpha * log_pis - min_q_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Compute alpha loss, only makes sense when not fixed obviously
            alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean() 
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            print("needs to be done, please dont fix alpha")
            raise NotImplementedError("Dont fix alpha")
            
            # soft update target networks
        self.soft_update(self.critic1, self.critic1_target, TAU)
        self.soft_update(self.critic2, self.critic2_target, TAU)
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), alpha_loss.item()

    def soft_update(self, used_model, target_model, tau):
        """ θ_target = τ*θ_local + (1 - τ)*θ_target (not in paper)

        Args:
            used_model: current model where weights will be copied from
            target_model: target model where weights will be copied to
            tau: balancing factor
        """
        for target_param, used_model_param in zip(target_model.parameters(), used_model.parameters()):
            target_param.data.copy_(tau*used_model_param.data + (1.0-tau)*target_param.data)