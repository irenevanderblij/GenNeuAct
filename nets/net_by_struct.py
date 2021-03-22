import collections
import torch.nn as nn
from nets.ActNet import ActNet

# Create sequential network by given structure. Structure is a list of sizes starting with the input size, 
# followed by the sizes of the hidden layers, and finally the output size.
def net_by_struct(net_struct):
    layers = collections.OrderedDict()
    
    for i in range(len(net_struct) - 2):
        layers['lin{}'.format(i)] = nn.Linear(net_struct[i], net_struct[i + 1])
        layers['relu{}'.format(i)] = nn.ReLU()
        
    layers['lin{}'.format(i+1)] = nn.Linear(net_struct[i+1], net_struct[i + 2])
    layers['logsoftmax'] = nn.LogSoftmax(dim=1)
    
    net = nn.Sequential(layers)
    
    return net

# Create ActNet network by given structure. Structure is a list of sizes starting with the input size, 
# followed by the sizes of the hidden layers, and finally the output size.
def net_by_struct_act(net_struct):
    layers = collections.OrderedDict()
    
    for i in range(len(net_struct) - 2):
        layers['lin{}'.format(i)] = nn.Linear(net_struct[i], net_struct[i + 1])
        layers['relu{}'.format(i)] = nn.ReLU()
        
    layers['lin{}'.format(i+1)] = nn.Linear(net_struct[i+1], net_struct[i + 2])
    layers['logsoftmax'] = nn.LogSoftmax(dim=1)
    
    net = ActNet(layers)
    
    return net