from torch.nn.modules.loss import NLLLoss
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import torch.nn as nn
import torch
from statsmodels.nonparametric.bandwidths import bw_silverman
import flatten_dict as fd
import matplotlib.lines as mlines
import pickle

# @weak_module
class NeuronLoss(NLLLoss):
    # The neural network
    global net
    
    # Init attributes
    activations_train = {}  # Train activations
    activations_test = {}   # Test activations
    
    train_kernels = {}  # Estimated density functions of training data
    hidden_neurons = [] # list of layer sizes
    
    layer_dict = {}     # Matching methods with their name
    neuron_memory = {}  # Storage variabloe used to retreive neuron activation values from a datapoint
    
    __constants__ = ['ignore_index', 'weight', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(NeuronLoss, self).__init__(weight, reduce, reduction=reduction)
        self.ignore_index = ignore_index
           

    # @weak_script_method
    def forward(self, inp, target):
        or_loss = F.nll_loss(inp, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return torch.FloatTensor(or_loss)
    
    # Set the net
    def set_net(self, actnet):
        self.net = actnet
    
    # Set the hidden layer size and fill the layer dictionary
    def set_hidden_layers(self, hidden_layers):
        self.hidden_neurons = hidden_layers
        
        # Init layer_dict
        for idx, (name, layer) in enumerate(self.net._modules.items()):
            self.layer_dict[layer] = (idx, name) 
    
    # Get the activation values from the training data and create the estimated density functions and store them     
    def set_activations(self, act_vis, act_vis_test, kernels=None, kernels_path=None):
        self.activations_train = act_vis
        self.activations_test = act_vis_test
        
        if kernels != None:
            self.train_kernels = kernels
        else:
        
            # Init train_kernals
            for layer in range(len(self.activations_train)):                      # For each hidden layer
                self.train_kernels[layer] = {}
                for neuron in range(len(self.activations_train[layer])):               # For each neuron in the hidden layer
                    self.train_kernels[layer][neuron] = {}
                    flat_values = [y for x in fd.flatten(self.activations_train[layer][neuron]).values() for y in x]
                    bandwidth = bw_silverman(flat_values, None)
                    if bandwidth == 0: bandwidth = 0.5 
                    
                    for output_class in range(len(self.activations_train[layer][neuron])):          # For each outputclass
                        train_data = np.array(self.activations_train[layer][neuron][output_class])
                        if train_data.shape[0] < 1: 
                            self.train_kernels[layer][neuron][output_class] = 0
                            continue
                        self.train_kernels[layer][neuron][output_class] = KernelDensity(kernel="gaussian", bandwidth=bandwidth, rtol=0.01).fit(train_data.reshape(-1,1))
            #Save kernels    
            if kernels_path != None:
                with open(kernels_path, 'wb+') as f:
                    pickle.dump(self.train_kernels, f, pickle.HIGHEST_PROTOCOL)
    
    # Get the fitness of the data
    def neuron_fitness(self, x, y):
        x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
        
        # Retrieve the activation values of the data
        new_act_vis = self.get_neuron_activations(x)
        self.activations_test = new_act_vis
        
        # Get the scores for the new data en activation values
        act_scores = self.get_act_scores(x)

        # Get the class probabilties
        class_probs = self.get_class_probs(act_scores)
        
        # Calculate loss 
        loss = torch.empty((len(class_probs)))
        for i, (pnt, pnt_y) in enumerate(zip(class_probs,y)):
            cur_list = pnt
            actual = cur_list[pnt_y]
            cur_list = np.delete(cur_list, pnt_y)
            best = min(cur_list)
            loss[i] = actual - best

        return loss
    
    # Score the neuron activation values to the training data
    def get_act_scores(self, x):
        
        num_pts = x.shape[0]
        act_scores = {}
        
        # Initialize scores dictionaries
        for pnt in range(num_pts):
            act_scores[pnt] = {}
        
            for layer in range(len(self.hidden_neurons)): 
                act_scores[pnt][layer] = {}  
                
                for neuron in range(self.hidden_neurons[layer]):               # For each neuron in the hidden layer
                    act_scores[pnt][layer][neuron] = {}
        
        # Fill the scores dictionary            
        for layer in range(len(self.hidden_neurons)):                       # For every layer
            for neuron in range(self.hidden_neurons[layer]):                # For every neuron
                act_test_x = list(self.activations_test[layer][neuron].values())

                for label in range(len(self.activations_train[layer][neuron])):    # For each output label
                    
                    # Retrieve train kernal belonging to this neuron and output class and score the current (test)data on this density function
                    kde_train = self.train_kernels[layer][neuron][label]
                    if kde_train == 0: 
                        score_exp = [0] * num_pts
                    else:
                        score = kde_train.score_samples(np.array(act_test_x).reshape(-1,1))
                        score_exp = np.exp(score)
                        
                    # Store results
                    for pnt in range(num_pts):
                        act_scores[pnt][layer][neuron][label] = score_exp[pnt]
        return act_scores 


    
    # Get the neuron activation values from specific data
    def get_neuron_activations(self, data_x):        
        # Add hooks to the network to be able to store neuron values
        self.add_hooks()
        num_pts = data_x.shape[0]
        
        # Create data structures for storing neuron activations
        act_vis_test = {}
        
        # Initialize dictionaries for storing activation values
        for i in range(len(self.hidden_neurons)):          # number of hidden layers
                act_vis_test[i] = {}
                for j in range(self.hidden_neurons[i]):        # number of neurons in this hidden layer
                    act_vis_test[i][j] = {}
            
        # Put data through the network
        _ = self.net(data_x)
        
        # Get the activation values from the neuron memory and store them
        for pnt in range(num_pts):
            for i in range(len(self.hidden_neurons)):              # number of hidden layers
                layer_neuron_activation = self.net.neuron_memory_batch[pnt][i*2 + 1]  # list of activation values for whole layer (all points), only activation layers
                for j in range(self.hidden_neurons[i]):            # number of neurons in this hidden layer
                    current_neuron_activation = layer_neuron_activation[j].item()
                    act_vis_test[i][j][pnt] = current_neuron_activation
                
        return act_vis_test
    
    # Calculate the estimated class probabilities from the scored activation values
    def get_class_probs(self, act_scores):
        num_pts = len(act_scores)
        output_size = len(act_scores[0][0][0])
        probs = np.empty((num_pts, output_size))
        
        for pnt in range(num_pts):
            dp_probs = [1] * output_size
            for layer in range(len(act_scores[pnt])):
                for neuron in range(len(act_scores[pnt][layer])):
                    neuron_probs = []
                    for label in range(len(act_scores[pnt][layer][neuron])):
                        current_prob = act_scores[pnt][layer][neuron][label]
                        if current_prob < 0.001: current_prob = 0.001       # Make sure the probabilities cannot become 0
                        neuron_probs = neuron_probs + [current_prob]
                    
                    neuron_probs_norm = [i/sum(neuron_probs) for i in neuron_probs] # Normalize

                    dp_probs = [i*j for i, j in zip(dp_probs,neuron_probs_norm)]    # Multiply probabilities

            dp_probs_norm = [i/sum(dp_probs) for i in dp_probs]     # Normalize
            dp_probs_norm = dp_probs
            dp_probs_norm = [1e-320 if i < 1e-320 else i for i in dp_probs_norm]    # change 0 to 1e-320
            dp_probs_norm_log = [math.log(i,2)/sum([math.log(x,2) for x in dp_probs_norm]) for i in dp_probs_norm]      # Take the log to make other classes also likely
            
            probs[pnt] = dp_probs_norm_log  
        return probs
    

    def plot_adv(self, test_x, test_y):  
        act_scores = {}
        
        for layer in range(len(self.activations_train)):   
            act_scores[layer] = {}                   # For each hidden layer
            fig, axes = plt.subplots(len(self.activations_train[layer]), 1) # len(self.activations_train[layer][0])
            fig.tight_layout()
            colors = ['co', 'ro', 'go']
            co = ['blue', 'red', 'green', 'purple', 'yellow', 'brown', 'pink', 'grey', 'lightblue', 'black']

            
            for neuron in range(len(self.activations_train[layer])):               # For each neuron in the hidden layer
                
                plt.figure()
                plt.xlabel("Neuron activation value")
                plt.ylabel("Density")
                
                act_scores[layer][neuron] = {}
                flat_values_train = [y for x in fd.flatten(self.activations_train[layer][neuron]).values() for y in x]
                flat_values_test = self.activations_test[layer][neuron]
                flat_values_test_y = test_y
                
                max_x = max(max(flat_values_train), max(flat_values_test))
                
                for label in range(len(self.activations_train[layer][neuron])):    # For each output label
                    
                    act_test_x = flat_values_test
                    
                    kde_train = self.train_kernels[layer][neuron][label]
                    
                    xrange = np.linspace(0, max_x, 1000)
                    kernal_train = kde_train.score_samples(xrange.reshape(-1,1))
                    
                    axes[neuron].plot(xrange, np.exp(kernal_train), label="KDE neuron (%i,%i) class %i" % (layer, neuron, label))
                    
                    plt.plot(xrange, np.exp(kernal_train), c=co[label])
                    
                    l = []
                    for dp in act_test_x:
                        score = kde_train.score(np.array(dp).reshape(-1,1))
                        l = l + [np.exp(score)]
                    for idx, val in enumerate(act_test_x):
                        marker = 'o' if flat_values_test_y[idx] == label else '.'
                        plt.plot(act_test_x[idx], l[idx], co[label], marker=marker)
                        plt.axvspan(val-0.001*max_x, val+.001*max_x, facecolor=co[flat_values_test_y[idx]])
                        print(idx, act_test_x[idx], l[idx])
                    score = kde_train.score(np.array(act_test_x).reshape(-1,1))
                    act_scores[layer][neuron][label] = l
                
                plt.draw()
                
                fig.text(0.5, 0.03, 'Neuron activation value', ha='center')
                fig.text(0.03, 0.5, 'Density', va='center', rotation='vertical')

                cox = ['blue', 'red', 'purple', 'green', 'yellow', 'brown', 'pink', 'grey', 'lightblue', 'black']
                handles = []
                for idx, c in enumerate(cox):
                    handles = handles + [mlines.Line2D([], [], color=c, label=idx)]

                plt.legend(handles=handles)                                                                                      
                plt.draw()
        return act_scores     
    
    # Method to add hooks to the model so that internal node values can be gathered
    def add_hooks(self):
        for name, layer in self.net._modules.items():
        #If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                self.add_hooks(layer)
            else:
                # it's a non sequential. Register a hook
                layer.register_forward_hook(self.hook_neurons)
    
    def hook_neurons(self, layer, input, output):
        (layer_ind, layer_name) = self.layer_dict[layer]
        self.neuron_memory[layer_ind] = output[0]
