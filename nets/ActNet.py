import torch.nn as nn
from collections import OrderedDict
import torchvision

# Class for a neural network with a neuron memory to retreive activation values
# @weak_module
class ActNet(nn.Sequential):
    
    layers = {}
    
    activations = {}
    neuron_memory = {}
    layer_dict = {}
    
    neuron_memory_batch = {}
    
    hook_handles = []
    
    def __init__(self, *args):
        super(nn.Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
            self.layers = [module for key, module in args[0].items()]
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
            self.layers = [module for key, module in self._modules.items()]
                        
        # Activation values init
        for idx, (name, layer) in enumerate(self._modules.items()):
            self.layer_dict[layer] = (idx, name)
            
        
        for i in range(int((len(self) / 2))):
            self.activations[i] = {}     # (ReLU) activations for hidden layer i
            
        self.hook_handles = []
        if self.training == False: 
            self.add_hooks(self)

    # If the network is loaded from file, we still need some initiating
    def init2(self):
        # Activation values init
        for idx, (name, layer) in enumerate(self._modules.items()):
            self.layer_dict[layer] = (idx, name)
        
        self.hook_handles = []
        if self.training == False: 
            self.add_hooks(self)
        
        for i in range(int((len(self) / 2))):
            self.activations[i] = {}     # (ReLU) activations for hidden layer i
    
    # Put an input through the network
    # @weak_script_method
    def forward(self, inp):
        for i in range(inp.shape[0]): # for each data point
            self.neuron_memory_batch[i] = {}
        for module in self._modules.values():
            if len(inp.shape) > 2: 
                inp = inp.view(inp.shape[0], -1) 
            inp = module(inp)
        return inp
    
    # Init activation memory for ReLU layers
    def hook_neurons(self, layer, inp, output):
        (layer_ind, _) = self.layer_dict[layer]
        self.neuron_memory[layer_ind] = output 
        
        for pnt in range(output.shape[0]):
            self.neuron_memory_batch[pnt][layer_ind] = output[pnt]
        
        
        if isinstance(layer, nn.ReLU):                    # ReLU layer
            relu_idx = int((layer_ind - 1) / 2)
            for neuron_idx, act in enumerate(output):
                self.activations[relu_idx][neuron_idx] = act
             
    
    # Method to add hooks to the model so that internal node values can be gathered
    def add_hooks(self):
        for _, layer in self._modules.items():
        #If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                self.add_hooks(layer)
            elif isinstance(layer, torchvision.models.resnet.Bottleneck):
                self.add_hooks(layer)
            else:
                # it's a non sequential. Register a hook
                handle = layer.register_forward_hook(self.hook_neurons)
                self.hook_handles = self.hook_handles + [handle]
    
    # Remove all hooks from the network         
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    # Training state
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.clear_hooks()
        
        return self

    # Eval state
    def eval(self, mode=False):
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.add_hooks()
        
        return self
                