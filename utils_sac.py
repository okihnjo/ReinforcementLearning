import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime, date
now = datetime.now()
current_time = now.strftime("%H.%M.%S")
today = date.today()

# dd/mm/YY
d1 = today.strftime("%d.%m.%Y")

if not os.path.exists(f"{d1}/{current_time}"):
    os.makedirs(f"{d1}/{current_time}/images")
    os.makedirs(f"{d1}/{current_time}/results")

# Hyperparameter
def initialize_weights(model: nn.Linear): 
    """Initializing weights for our linear model networks.

    Args:
        m (nn.Linear): The model we want to initialize its weights
    """
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight, gain=1)
        torch.nn.init.constant_(model.bias, 0)


def save_network(model: nn.Linear):
    torch.save(model.state_dict(), f"{d1}/{current_time}/trained_pendu.pt")

def hidden_init(layer):
    """he Xavier initialization method assumes that the activations of the layer should have a variance of 1. This assumption helps in preventing the activations from vanishing or exploding during training.
The variance of the activations in a fully connected layer is influenced by the number of input units or the fan-in of the layer. Intuitively, a larger fan-in requires smaller weights to prevent the variance from exploding, while a smaller fan-in requires larger weights to prevent the variance from vanishing.
By setting lim as the reciprocal of the square root of the fan-in, i.e., lim = 1. / np.sqrt(fan_in), the weights are initialized within a range that takes into account the number of input units. The reciprocal of the square root ensures that larger fan-in values result in smaller weight limits, while smaller fan-in values result in larger weight limits.
The weights are then randomly initialized within the range (-lim, lim) to ensure that they are initially distributed around zero.

    Args:
        layer (_type_): _description_

    Returns:
        _type_: _description_
    """
    fan_in = layer.weight.data.size()[0] # number of input_layer
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def moving_mean(losses: tuple):
    fig = go.Figure()
    for i in range(len(losses)):
        window_size = 10000
        num_segments = len(losses[i]) // window_size
        segments = np.array_split(losses[i], num_segments)
        averages = [segment.mean() for segment in segments]
        
        fig.add_trace(go.Scatter(x=[x for x in range(len(averages))], y=averages, mode="markers"))
    fig.show()
    go.Figure.write_image(fig, f"{d1}/{current_time}/images/losses.png")

def load_model(model: nn.Linear, path: str):
    model.load_state_dict(torch.load(path))

def plot_reward(rew: list):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x in range(len(rew))], y=rew, mode="markers"))
    fig.show()
    go.Figure.write_image(fig, f"{d1}/{current_time}/images/reward.png")


