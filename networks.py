import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils_sac import xavier_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
TAU = 1e-2
HIDDEN_SIZE = 256
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256        # minibatch size
LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 5e-4       # learning rate of the critic
FIXED_ALPHA = None

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.init_w = init_w
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*xavier_init(self.fc1))
        self.fc2.weight.data.uniform_(*xavier_init(self.fc2))
        self.mu.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)

    def forward(self, state):

        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
 
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # to obtain actions, we sample a z-value from the obtained Gaussian distribution
        # later, we will take the hyperbolic tangent of the z value to obtain our action. 
        z = normal.rsample()

        # we modify the log_pi computation as explained in the Haarnoja et al. paper
        log_pi = (normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)).sum(1, keepdim=True)
        next_action = torch.tanh(z)
        return next_action, log_pi

        
    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*xavier_init(self.fc1))
        self.fc2.weight.data.uniform_(*xavier_init(self.fc2))
        self.fc3.weight.data.uniform_(-4e-3, 4e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
