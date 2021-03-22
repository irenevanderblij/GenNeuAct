import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torchvision
import numpy as np
from method import adversarial_methods
import matplotlib as mpl
from matplotlib.patches import Patch
from nets.net_by_struct import net_by_struct_act
from loss.neuron_loss import NeuronLoss
import os
import dill
from scipy.spatial import distance
from method.train_adversarial import  train_adversarial_x
from sklearn import preprocessing
from method import cwbase
import random
from datasets.DiabetesDataset import DiabetesDataset
from loss.neuron_loss_hist import NeuronLossHist
from sklearn.model_selection import train_test_split
from datasets.BreastCancerDataset import BreastCancerDataset
from datasets.DigitsDataset import DigitsDataset
from datasets.IrisDataset import IrisDataset
from data.load_data import load_data
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Global variables
global criterion
global net
global optimizer

# Get the neuron activations from a train_loader with a certain training method
def get_neuron_activations2_dl(train_loader, adv_method):
        
    # Create data structures for storing neuron activations
    act_vis = {}
    pred_y = {}
    for out in range(output_size):
        pred_y[out] = []
    
    # Init data structure
    for i in range(len(net_struct)-2):          # number of hidden layers
        act_vis[i] = {}
        for j in range(net_struct[i+1]):        # number of neurons in this hidden layer
            act_vis[i][j] = {}
            for k in range(net_struct[len(net_struct)-1]):      # for each output class
                act_vis[i][j][k] = []
    
    for _, (data_x, data_y) in enumerate(train_loader):
        data_x = data_x.view(data_x.shape[0], -1)
    
        pred_y_data_format = []
        
        for idx, (x,y) in enumerate(zip(data_x, data_y)):
            if random.uniform(0,1) < train_adversarial_percentage:
                x = adversarial_methods.create_adversarial_example(torch.stack([x], dim=0), torch.stack([y], dim=0), net, criterion, method=adv_method, dataset=data_x)
            else: x = torch.stack([x], dim=0)
            out = net(x)
            
            _, pred = torch.max(out, 1)
            pred_y[y.item()].append(pred.data)
            pred_y_data_format.append(pred.data.item())
            
            for i in range(len(net_struct)-2):              # number of hidden layers
                layer_neuron_activation = net.neuron_memory[i*2 + 1][0]  # list of activation values for whole layer
                for j in range(net_struct[i+1]):            # number of neurons in this hidden layer
                    current_neuron_activation = layer_neuron_activation[j].item()
                    act_vis[i][j][y.item()].append(current_neuron_activation)
    
    return act_vis, pred_y

def get_neuron_activations_testdata(data_x):
    global neuron_memory
    neuron_memory = {}
    
    # Add hooks to the network to be able to store neuron values
    net.add_hooks(net)
    
    # Create data structures for storing neuron activations
    act_vis_test = {}
    pred_y = []
    
    for i in range(len(net_struct)-2):          # number of hidden layers
        act_vis_test[i] = {}
        for j in range(net_struct[i+1]):        # number of neurons in this hidden layer
            act_vis_test[i][j] = []
    
    for _, x in enumerate(data_x):

        out = net(torch.stack([x], dim=0))
        
        _, pred = torch.max(out, 1)
        pred_y.append(pred.data)
        
        for i in range(len(net_struct)-2):              # number of hidden layers
            layer_neuron_activation = neuron_memory[i*2 + 1]  # list of activation values for whole layer
            for j in range(net_struct[i+1]):            # number of neurons in this hidden layer
                current_neuron_activation = layer_neuron_activation[j].item()
                act_vis_test[i][j].append(current_neuron_activation)
    
    return act_vis_test, pred_y

def get_neuron_activations_testdata_dl(test_loader):
    
    # Create data structures for storing neuron activations
    act_vis_test = {}
    pred_y = []
    
    for i in range(len(net_struct)-2):          # number of hidden layers
        act_vis_test[i] = {}
        for j in range(net_struct[i+1]):        # number of neurons in this hidden layer
            act_vis_test[i][j] = []
    
    for _, (test_x, _) in enumerate(test_loader):
        test_x = test_x.view(test_x.shape[0], -1)
        
        for _, x in enumerate(test_x):
    
            out = net(torch.stack([x], dim=0))
            
            _, pred = torch.max(out, 1)
            pred_y.append(pred.data)
            
            for i in range(len(net_struct)-2):              # number of hidden layers
                layer_neuron_activation = net.neuron_memory[i*2 + 1][0]  # list of activation values for whole layer
                for j in range(net_struct[i+1]):            # number of neurons in this hidden layer
                    current_neuron_activation = layer_neuron_activation[j].item()
                    act_vis_test[i][j].append(current_neuron_activation)
    
    return act_vis_test, pred_y


def plot_vis(act_vis, pred_y, layer_idx=0):
    layer_data = act_vis[layer_idx]

    num_neurons = net_struct[layer_idx+1]
    
    width_ratio = []
    for key in layer_data[0]:
        width_ratio.append(len(layer_data[0][key]))
    
    rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labeltop" : False,
      "xtick.top" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False,
      "figure.subplot.hspace" : 0.1,
      "figure.constrained_layout.hspace" : 0}
    plt.rcParams.update(rc)
    fig, axes = plt.subplots(2 * num_neurons, output_size, sharex = False, sharey= False, gridspec_kw={'height_ratios': [4,1] * num_neurons, 'width_ratios': width_ratio}) # , gridspec_kw={'height_ratios': [4, 1,4,1,4,1]}
    colors = np.array([[1.0, 0.482, 0.0, 1], [1.0,0.0,0.616, 1], [0.0, 0.533, 1.0, 1]]) #orange, pink, blue
    colors_capped = colors[0:output_size]
    cmapx = mpl.colors.ListedColormap(colors_capped)
    
    for k in range(num_neurons):
        flat_values = [y for x in fd.flatten(layer_data[k]).values() for y in x]
        max_d = max(flat_values) + 0.0001
        min_d = min(flat_values)
        
        for j in range(output_size):            
            ax = axes[k * 2, j] #if num_neurons > 1 else axes[j]
            
            c = ax.pcolor(np.expand_dims(layer_data[k][j],0), vmin=min_d, vmax=max_d)
            ax.set_title("Class %i" %int(j), pad=-20, color='#ffffff')
            
            ax2 = axes[k * 2 + 1, j] #if num_neurons > 1 else axes[j + 1]
            
            ax2.matshow(np.expand_dims(pred_y[j],0), cmap=cmapx, vmin=0, vmax=output_size-1, extent=[0,len(pred_y[j]),0,1])
            ax2.tick_params(axis=u'x', top=False, bottom=False)
            ax2.set_aspect('auto')
            
            if k == (num_neurons - 1): 
                ax2.tick_params(axis=u'x', which=u'both',length=3, bottom=True, labelbottom=True)
                sum_ind = 0
                for prev in range(j): sum_ind = sum_ind + len(pred_y[prev])
                plt.setp(ax2, xticks=np.arange(0, len(pred_y[j]), 100), xticklabels=np.arange(sum_ind, len(pred_y[j]) + sum_ind, 100)) 
            
            if j == 0:
                plt.setp(ax, yticks=[0.8], yticklabels=["Neuron (%i,%i)" % (layer_idx, k)])
                ax.tick_params(axis=u'y', labelleft=True, labelrotation=90)
                
            if j == output_size - 1:
                prev_neuron_axes = []
                for ind in range(j): prev_neuron_axes.extend([axes[k * 2, ind], axes[k * 2 + 1, ind]])
                fig.colorbar(c, ax=[ax, ax2] + prev_neuron_axes, shrink=0.9)

    handles = [Patch(facecolor=colors[0], label='Class 0'), Patch(facecolor=colors[1], label='Class 1'), Patch(facecolor=colors[2], label='Class 2')]
    fig.legend(handles=handles[0:output_size], bbox_to_anchor=(0.43,0.925), loc='center', ncol=output_size)
    
#     cbar = fig.colorbar(c, ax=axes.ravel().tolist())
#     cbar.ax.set_ylabel('Neuron activation value', rotation=90)

    fig.text(0.45, 0.03, 'Test samples sorted by true label', ha='center')
    plt.draw()
 
def create_advs(data_x, data_y, adv_method):
    advs = data_x.clone()
    for idx, (x,y) in enumerate(zip(data_x, data_y)):
        adv_x = adversarial_methods.create_adversarial_example(torch.stack([x], dim=0), torch.stack([y], dim=0), net, criterion, method=adv_method, dataset=data_x, epsilon=learning_rate)
        advs[idx] = adv_x[0]
        if idx % 10 == 0:
            print("Generate adv", adv_method, ":", idx, "/", len(data_x))
    return advs

# Calculate accuracy scores
def score_advs(x, y, adv_method):
    predict_out = net(x)
    _, predict_y = torch.max(predict_out, 1)
    return predict_y

# SETTIN

test_size_abs, batch_size_train, batch_size_test, hidden_layers, modelname, hist, hist_str = 0, 0, 0, [], "", False, ""

#### SETTINGS ####
train_adversarial_percentage = 0.75
learning_rate = 0.05
epochs = 1000
train_method = 'natural'

save_model = True
save_every_batch = True

load_network = False
load_activations = False
load_advs = False

name_dataset = 'iris'


# Load data
if name_dataset == "fashion":
    modelname = "natural_91"
    modelname_pgd = "pgd_14"
    batch_size_train = 100
    batch_size_test = 100
    test_size_abs = 10000
    hidden_layers = [100,100]
    hist = True
    hist_str = "_hist" if hist else ""
    
    test_data = torchvision.datasets.FashionMNIST('/mnist/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor() ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    train_data = torchvision.datasets.FashionMNIST('/mnist/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor() ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=False)
  
    input_size = 28 * 28
    output_size = 10
    
elif name_dataset == "diabetes":
    modelname = "natural_5"
    modelname_pgd = "pgd_5"
    
    batch_size_train = 10
    batch_size_test = 10
    test_size_abs = 77
    hidden_layers = [20,20]
    hist = False
    hist_str = "_hist" if hist else ""
    
    dataset = load_data(name_dataset)
    
    #Normalize data between 0-1
    atts = dataset.columns[0:dataset.shape[1]-1]
    dataset[atts] = preprocessing.normalize(dataset[atts]) 

    if os.path.isfile("../data/splits/" + name_dataset + "_trainsplit.pkl"):
        print("Split loading")
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'rb') as f:
            train = pickle.load(f)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'rb') as f:
            test = pickle.load(f)
    else:
        train, test =  train_test_split(dataset, test_size=0.1, shuffle=True, stratify=dataset['Outcome'], random_state=12)
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'wb+') as f:
            pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'wb+') as f:
            pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
    
    train_data = DiabetesDataset(train, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    test_data = DiabetesDataset(test, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)
        
    # Create network based on net_struct
    net_struct = [dataset.shape[1] - 1] + hidden_layers + [dataset[dataset.columns[dataset.shape[1]-1]].nunique()]
    input_size = net_struct[0] # example_data.shape[1] * example_data.shape[2] * example_data.shape[3]
    output_size = net_struct[len(net_struct)-1]
    
elif name_dataset == "breastcancer":
    modelname = "natural_3"
    modelname_pgd = "pgd_3"
    batch_size_train = 10
    batch_size_test = 10
    test_size_abs = 100
    hidden_layers = [15,15]
    hist = False
    hist_str = "_hist" if hist else ""

    dataset = pd.read_csv("D:/Users/Irene/workspace/MC_iris/data/breastcancer.csv")
    x = dataset.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
    x[:] = preprocessing.normalize(x[:]) 
    diag = { "M": 1, "B": 0}
    y = dataset["diagnosis"].replace(diag)
    dataset = x.join(y)
    
    if os.path.isfile("../data/splits/" + name_dataset + "_trainsplit.pkl"):
        print("Split loading")
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'rb') as f:
            train = pickle.load(f)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'rb') as f:
            test = pickle.load(f)
    else:
        train, test =  train_test_split(dataset, test_size=100, shuffle=True, stratify=dataset['diagnosis'], random_state=12)
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'wb+') as f:
            pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'wb+') as f:
            pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
    
    train_data = BreastCancerDataset(train, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    test_data = BreastCancerDataset(test, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    
    input_size = dataset.shape[1] - 1
    output_size = dataset[dataset.columns[dataset.shape[1]-1]].nunique()

elif name_dataset == "digits":
    modelname = "natural_2"
    modelname_pgd = "pgd_2"
    
    batch_size_train = 10
    batch_size_test = 50
    test_size_abs = 1500
    hidden_layers = [25,25]
    hist = True
    hist_str = "_hist" if hist else ""
    
    dataset = pd.read_csv("D:/Users/Irene/workspace/MC_iris/data/" + name_dataset + ".csv")
    
    #Normalize data between 0-1
    atts = dataset.columns[0:dataset.shape[1]-1]
    dataset[atts] = preprocessing.normalize(dataset[atts]) 
    
    if os.path.isfile("./data/splits/" + name_dataset + "_trainsplit.pkl"):
        print("Split loading")
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'rb') as f:
            train = pickle.load(f)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'rb') as f:
            test = pickle.load(f)
    else:
        train, test =  train_test_split(dataset, test_size=1500, shuffle=True, stratify=dataset['class'], random_state=12)
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'wb+') as f:
            pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        with open("../data/splits" + name_dataset + "_testsplit.pkl", 'wb+') as f:
            pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)    
            
    train_data = DigitsDataset(train, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    test_data = DigitsDataset(test, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    
    input_size = dataset.shape[1] - 1
    output_size = dataset[dataset.columns[dataset.shape[1]-1]].nunique()

elif name_dataset == "iris":
    modelname = "natural"
    modelname_pgd = "pgd"
    batch_size_train = 5
    batch_size_test = 30
    test_size_abs = 30
    hidden_layers = [2,3,2]
    hist = False
    hist_str = "_hist" if hist else ""
    
    dataset = pd.read_csv('D:/Users/Irene/Documents/School/Master/Thesis/data/iris2.csv')
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2
    
    atts = dataset.columns[0:dataset.shape[1]-1]
    dataset[atts] = preprocessing.normalize(dataset[atts]) 
    
    if os.path.isfile("../data/splits/" + name_dataset + "_trainsplit.pkl"):
        print("Split loading")
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'rb') as f:
            train = pickle.load(f)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'rb') as f:
            test = pickle.load(f)
    else:
        train, test =  train_test_split(dataset, test_size=.2, shuffle=True, stratify=dataset['species'], random_state=12)
        with open("../data/splits/" + name_dataset + "_trainsplit.pkl", 'wb+') as f:
            pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        with open("../data/splits/" + name_dataset + "_testsplit.pkl", 'wb+') as f:
            pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
    
    train_data = IrisDataset(train, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    test_data = IrisDataset(test, transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    
    input_size = dataset.shape[1] - 1
    output_size = dataset[dataset.columns[dataset.shape[1]-1]].nunique()
    
# Array with the network structure

net_struct = [input_size] + hidden_layers + [output_size]

train_method = 'ta_pgd' if hist else 'natural'

load_network_name = "./example_nets/digits_0.pt"
# load_network_name = "./nets/" + name_dataset + "/" + modelname + ".pt"
activations_train = "./activations/" + name_dataset + "_" + modelname + "_trainact.pkl"
activations_test = "./activations/" + name_dataset + "_" + modelname + "_testact.pkl"
activations_kernel = "./activations/" + name_dataset + "_" + modelname + "_kernelact" + hist_str + ".pkl"

advs_x_file = "./advs/"+ name_dataset + "_x_" + modelname + "_{}_{}" + hist_str + ".npy"
advs_y_file = "./advs/"+ name_dataset + "_y_" + modelname + "_{}_{}" + hist_str + ".npy"
advs_cw_file = "./advs/"+ name_dataset + "_cw_" + modelname + "_{}_{}" + hist_str + ".npy"
advs_kde_file = "./advs/"+ name_dataset + "_kde_" + modelname + "_{}_{}" + hist_str + ".npy"


if not load_network: 
    # Create network from scratch
    net = net_by_struct_act(net_struct)
    
    if hist:
        criterion = NeuronLossHist(reduction='none')
    else:
        criterion = NeuronLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    criterion.set_net(net)
    criterion.set_hidden_layers(hidden_layers)
    
    # Train network
    net = train_adversarial_x(train_data, net, optimizer, criterion, epochs=epochs, train_method=train_method, train_adversarial_percentage=train_adversarial_percentage)
    
    criterion.set_net(net)
else: 
    # Load stored net
    net = torch.load(load_network_name, map_location=lambda storage, loc: storage, pickle_module=dill)
    net.init2()
    if hist:
        criterion = NeuronLossHist(reduction='none')
    else:
        criterion = NeuronLoss(reduction='none')
    criterion.set_net(net)
    criterion.set_hidden_layers(hidden_layers)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Print accuracy scores for train and test set
all_x_train = torch.Tensor(len(train_loader) * batch_size_train, input_size)
all_y_train = torch.LongTensor(len(train_loader) * batch_size_train)
all_x_test = torch.Tensor(test_size_abs, input_size)
all_y_test = torch.LongTensor(test_size_abs)
for batch_idx, (X_batch, y_batch) in enumerate(train_loader):    
    X_batch = X_batch.view(X_batch.shape[0], -1)
    all_x_train[batch_idx * batch_size_train:min(batch_idx * batch_size_train + batch_size_train, len(train_data))] = X_batch
    all_y_train[batch_idx * batch_size_train:min(batch_idx * batch_size_train + batch_size_train, len(train_data))] = y_batch
for batch_idx, (X_batch, y_batch) in enumerate(test_loader):    
    X_batch = X_batch.view(X_batch.shape[0], -1)
    all_x_test[batch_idx * batch_size_test:min(batch_idx * batch_size_test + batch_size_test, len(test_data))] = X_batch
    all_y_test[batch_idx * batch_size_test:min(batch_idx * batch_size_test + batch_size_test, len(test_data))] = y_batch
predict_out_train = net(all_x_train)
_, predict_y_train = torch.max(predict_out_train, 1)
print ('Train prediction accuracy:', accuracy_score(all_y_train.data, predict_y_train.data))
predict_out_test = net(all_x_test)
_, predict_y_test = torch.max(predict_out_test, 1)
print ('Test prediction accuracy:', accuracy_score(all_y_test.data, predict_y_test.data))

# Save network
if save_model: torch.save(net, "../nets/nets/model.pt", pickle_module=dill)   

# Done training, set net to evaluation mode and add hooks for storing neuron activation values
net.eval() 
# net.add_hooks(net)
kernels = None

# Load or create activations
if load_activations:
    print("Loading activations")
    with open(activations_train, 'rb') as f:
        act_vis_natural = pickle.load(f)
    with open(activations_test, 'rb') as f:
        act_vis_test_natural = pickle.load(f)
#     with open(activations_kernel, 'rb') as f:
#         kernels = pickle.load(f)
else:
    print("Creating activations")
    act_vis_natural, pred_natural = get_neuron_activations2_dl(train_loader, train_method)
    act_vis_test_natural, pred_test_natural = get_neuron_activations_testdata_dl(test_loader)
    
    with open(activations_train, 'wb+') as f:
        pickle.dump(act_vis_natural, f, pickle.HIGHEST_PROTOCOL)
    with open(activations_test, 'wb+') as f:
        pickle.dump(act_vis_test_natural, f, pickle.HIGHEST_PROTOCOL)
criterion.set_activations(act_vis_natural, act_vis_test_natural, kernels=kernels, kernels_path=activations_kernel)


# Helper variables (can be moved)
num_hidden_layers = len(net_struct) - 2
output_size = net_struct[len(net_struct)-1]

# Create adversarial examples batchwise for GenNeuAct and C&W
if load_advs == False:
    adversary = cwbase.L2Adversary(targeted=False, box=(0.,1.))
    for batch_idx, (test_X_batch, test_y_batch) in enumerate(test_loader):
#         if batch_idx * batch_size_test + batch_size_test <= start_limit: continue
        print("Creating advs, batch", batch_idx)
         
        data_x = test_X_batch.view(test_X_batch.shape[0], -1)
        advs_kde = adversarial_methods.create_adversarial_example(data_x, test_y_batch, net, criterion, method='genneuact')
        advs_cw = adversary(net, data_x, test_y_batch, to_numpy=False)

        if save_every_batch:
            batch_min = batch_idx * batch_size_test
            batch_max = min(batch_min + batch_size_test - 1, test_size_abs)
            np.save(advs_kde_file.format(batch_min, batch_max), advs_kde.numpy())
            np.save(advs_cw_file.format(batch_min, batch_max), advs_cw.numpy())
            np.save(advs_y_file.format(batch_min, batch_max), test_y_batch.numpy())
            np.save(advs_x_file.format(batch_min, batch_max), data_x.numpy())
        
        predict_kde = score_advs(advs_kde, test_y_batch, 'genneuact')
        print("kde succesfull: %i" % (sum(test_y_batch.numpy()!=predict_kde.numpy())))
        predict_cw = score_advs(advs_cw, test_y_batch, 'cw')
        print("cw succesfull: %i" % (sum(test_y_batch.numpy()!=predict_cw.numpy())))
        
advs_loaded = {}

# Combine batches
shape = all_x_test.shape
limit = len(test_loader)

all_x = torch.zeros(min(shape[0], limit*batch_size_test), shape[1])
all_y = torch.LongTensor(min(shape[0], limit*batch_size_test))
all_advs_kde = torch.zeros(min(shape[0], limit*batch_size_test), shape[1])
all_advs_cw = torch.zeros(min(shape[0], limit*batch_size_test), shape[1])

# Load x, y, GenNeuAct advs, CW advs
for batch_idx in range(limit): 
    batch_min = batch_idx * batch_size_test
    batch_max = min(batch_min + batch_size_test - 1, test_size_abs)
    m = min(batch_idx*batch_size_test+batch_size_test, shape[0])
    all_y[batch_idx*batch_size_test:m] = torch.from_numpy(np.load(advs_y_file.format(batch_min, batch_max)))
    all_x[batch_idx*batch_size_test:m] = torch.from_numpy(np.load(advs_x_file.format(batch_min, batch_max)))
    all_advs_cw[batch_idx*batch_size_test:m] = torch.from_numpy(np.load(advs_cw_file.format(batch_min, batch_max)))
    all_advs_kde[batch_idx*batch_size_test:m] = torch.from_numpy(np.load(advs_kde_file.format(batch_min, batch_max)))

advs_loaded['gen_kde_norm_batch'] = all_advs_kde
advs_loaded['cw'] = all_advs_cw

methods = ['gen_kde_norm_batch', 'cw', 'ta_fgsm', 'ta_bim', 'ta_rfgsm']
methods_dict = {'gen_kde_norm_batch' : 'GenNeuAct', 'cw' : 'C&W', 'ta_fgsm' : "FGSM", 'ta_bim' : 'BIM', 'ta_deepfool' : 'DeepFool', 'ta_rfgsm' : 'RFGSM', 'ta_cw' : 'C&W (default c)'}
advs_overview = np.zeros((len(methods), test_size_abs))
advs_distance = np.zeros((len(methods), test_size_abs))

all_x_imageview = all_x.view(min(shape[0], limit*batch_size_test), 1, shape[1], 1)



advs_all = {'original' : all_x_imageview}
for idx, method in enumerate(methods): 
    if method in advs_loaded:
        advs = advs_loaded[method]
        np.save("./advs/" + name_dataset + "_" + method + "_" + modelname + "_0_" + str(test_size_abs) + hist_str + ".npy", advs.numpy())
#     elif os.path.isfile("./advs/" + name_dataset + "/" + method + "_" + modelname + "_0_" + str(test_size_abs) + hist_str + ".npy"):
#         advs = torch.from_numpy(np.load("./advs/" + name_dataset + "/" + method + "_" + modelname + "_0_" + str(test_size_abs) + hist_str + ".npy"))
    else:
        advs = adversarial_methods.create_adversarial_example(all_x_imageview, all_y, net, criterion, method=method, optimizer=optimizer)
        np.save("./advs/" + name_dataset + "_" + method + "_" + modelname + "_0_" + str(test_size_abs) + hist_str + ".npy", advs.numpy())
    advs_all[method] = advs

    predict = score_advs(advs, all_y, method)
    advs_overview[idx] = all_y.numpy()!=predict.numpy()
    print(method, "succesfull: %i" % (sum(all_y.numpy()!=predict.numpy())))

    advs_distance[idx] = [distance.euclidean(u, v) for (u,v) in zip(advs,all_x_imageview)]

advs_overview_all = np.zeros((len(methods)+1, test_size_abs))
advs_overview_all[0] = (np.repeat(True, test_size_abs))
advs_overview_all[1:] = advs_overview

all_success = np.all(advs_overview == advs_overview[0,:], axis = 0)
   
mean = []
for i in range(len(advs_distance)):
    current = advs_distance[i]
    success = current[advs_overview[i]==1]
    mean_pnt = np.mean(success)
    mean = mean + [mean_pnt]
print("Mean distances:", mean)

# Plot adversarial examples by their success
colors = ['blue', 'darkorange', 'saddlebrown', 'purple', 'green', 'deepskyblue', 'gold']
rc = {"axes.spines.left" : True,
      "axes.spines.right" : False,
      "axes.spines.bottom" : True,
      "axes.spines.top" : False,
      "xtick.bottom" : True,
      "xtick.labeltop" : False,
      "xtick.top" : False,
      "xtick.labelbottom" : True,
      "ytick.labelleft" : True,
      "ytick.left" : True,
      "ytick.minor.left" : False,
      "figure.constrained_layout.hspace" : 0,
      "axes.labelsize" : 'medium',
      "axes.autolimit_mode" : "data",
      "axes.xmargin" : 0.01,
      "axes.ymargin" : 0.01}
plt.rcParams.update(rc)  
plt.figure(figsize=(10,5))
plt.imshow(advs_overview, cmap = mpl.colors.ListedColormap(['powderblue', 'darkorchid']))

plt.ylim(-.5, len(methods) )
plt.gca().set_yticks(np.arange(len(methods)))

plt.gca().set_yticklabels([methods_dict[method] for method in methods])
plt.gca().set_aspect(0.5)
plt.xlabel("Data points"); plt.ylabel("Adversarial method")
plt.gca().set_yticks(np.arange(len(methods))-0.5, minor=True)
# plt.gca().set_xticks(np.arange(77), minor=True)
plt.gca().grid(which='minor', color='w', linestyle='-', linewidth=1.5)
handles = [mlines.Line2D([], [], color='powderblue', marker = 's', linestyle='None', label="Unsuccessful adversarial example"),mlines.Line2D([], [], color='darkorchid', marker = 's', linestyle='None', label="Successful adversarial example") ]
plt.legend(handles=handles, bbox_to_anchor=(0.5, 1.5), loc='upper center')
plt.show()

# Load adversarial examples
def load_advs(name_dataset, modelname):
    load_network_name = "./nets/" + name_dataset + "/" + modelname + ".pt"
    net2 = torch.load(load_network_name, map_location=lambda storage, loc: storage, pickle_module=dill)
    net2.init2()
    if hist:
        criterion = NeuronLossHist(reduction='none')
    else:
        criterion = NeuronLoss(reduction='none')
    criterion.set_net(net2)
    criterion.set_hidden_layers(hidden_layers)
    optimizer = torch.optim.SGD(net2.parameters(), lr=learning_rate)
    
    advs_overview2 = np.zeros((len(methods), test_size_abs))
    advs_distance2 = np.zeros((len(methods), test_size_abs))
    for idx, method in enumerate(methods): 
        advs = torch.from_numpy(np.load("./advs/" + name_dataset + "/" + method + "_" + modelname + "_0_" + str(test_size_abs) + hist_str + ".npy"))
        predict_out = net2(advs)
        _, predict_y = torch.max(predict_out, 1)
        print(method, "succesfull: %i" % (sum(all_y.numpy()!=predict_y.numpy())))
        advs_overview2[idx] = all_y.numpy()!=predict_y.numpy()
        advs_distance2[idx] = [distance.euclidean(u, v) for (u,v) in zip(advs,all_x_imageview)]
    return advs_overview2, advs_distance2


# Run for different c values
def experiment_c_values():
    cs = np.logspace(-1,5, num=19)
    advs_overview_single = np.zeros((len(cs), test_size_abs))
    advs_distance_single = np.zeros((len(cs), test_size_abs))
    for idx, i in enumerate(cs):
        adversary = cwbase.L2Adversary(targeted=False, box=(0.,1.), c_range=(i, i+1e-3), search_steps=1)
        advs = adversary.attack2(net, all_x_imageview, all_y, to_numpy=False)
          
        predict = score_advs(advs, all_y, 'cw_'+ str(i))
        advs_overview_single[idx] = all_y.numpy()!=predict.numpy()
        print("cw_" + str(i), "succesfull: %i" % (sum(all_y.numpy()!=predict.numpy())))
        advs_distance_single[idx] = [distance.euclidean(u, v) for (u,v) in zip(advs,all_x_imageview)]
          
        with open("./nets/" + name_dataset + "/cwx" + str(i) + ".pkl", 'wb+') as f:
            pickle.dump(advs, f, pickle.HIGHEST_PROTOCOL)
                 
    mean = []
    for j in range(len(advs_distance_single)):   
        current = advs_distance_single[j]
        success = current[advs_overview_single[j]==1]
        mean_pnt = np.mean(success)
        mean = mean + [mean_pnt]
    print(mean)
    cs = np.logspace(-3,3, num=19)
    cs = np.append(cs, 2154.4347)
    advs_overview = np.zeros((len(cs), test_size_abs))
    advs_distance = np.zeros((len(cs), test_size_abs))
    for idx, method in enumerate(cs): 
        with open("./nets/fashion/cwx" + str("%.4f" % method) + ".pkl", 'rb') as f:
            advs = pickle.load(f)
    
        predict = score_advs(advs, all_y, method)
        advs_overview[idx] = all_y.numpy()!=predict.numpy()
        print(method, "succesfull: %i" % (sum(all_y.numpy()!=predict.numpy())))
        advs_distance[idx] = [distance.euclidean(u, v) for (u,v) in zip(advs,all_x_imageview)]
    
    mean = []
    for i in range(len(advs_distance)):
        current = advs_distance[i]
        success = current[advs_overview[i]==1]
        mean_pnt = np.mean(success)
        mean = mean + [mean_pnt]
    print(mean)


 


