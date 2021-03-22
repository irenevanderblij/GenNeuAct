import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable

# Load the iris or diabetes dataset
def load_data(dataset):
   
    if dataset == 'iris':
        return load_iris()
    elif dataset == 'diabetes':
        return load_diabetes()
    else:
        raise Exception('No such dataset')
    

def load_iris():
    dataset = pd.read_csv('./data/iris.csv')
    
    # transform species to numerics
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2
    
    return dataset

def load_diabetes():
    dataset = pd.read_csv("./data/diabetes.csv")
    
    return dataset

def testtrainsplit(dataset, test_size):
    data_size = dataset.shape
    
    # Split randomly with equal amount of samples per class
    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:data_size[1]-1]].values, dataset[dataset.columns[data_size[1]-1]].values, test_size=test_size, stratify=dataset[dataset.columns[data_size[1]-1]].values, random_state=1)

    train_y = train_y.astype(float)
    test_y = test_y.astype(float)
    
    # wrap up with Variable in pytorch
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())
    
    return train_X, test_X, train_y, test_y
    
    
    