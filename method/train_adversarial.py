import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
from method.adversarial_methods import create_adversarial_example
import random
import dill

# https://github.com/locuslab/fast_adversarial/blob/master/MNIST/train_mnist.py



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='mnist_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def train_adversarial(train_x, train_y, model, opt, criterion, batch_size=100, data_dir="", epochs=10, attack='fgsm', epsilon=0.3, alpha=0.375, attack_iters=40, lr_max=5e-3, lr_type="cyclic", fname="", seed=0):
    alpha = 0.1
    epsilon = 0.1
    
    np.random.seed(seed)
    torch.manual_seed(seed)

#     mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
#     train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
#     train_loader = torch.utils.data.DataLoader(zip(train_x, train_y), batch_size=batch_size, shuffle=True)

    model.train()
# 
#     opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    
    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')

#     criterion = nn.CrossEntropyLoss()

#     logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        for i, (X, y) in enumerate(zip(train_x, train_y)):
            X = torch.stack([X], dim=0)
            y = torch.stack([y], dim=0)

            lr = lr_schedule(epoch + (i+1)/len(train_x))
            opt.param_groups[0].update(lr=lr)

            if attack == 'fgsm':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to("cpu")
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                delta = delta.detach()
            elif attack == 'none':
                delta = torch.zeros_like(X)
            elif attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
                
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
        if epoch % 1 == 0:
            print("Training epoch", epoch, "/", epochs, "loss:", loss.item(), train_acc)

    train_time = time.time()
#         logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
#             epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
    torch.save(model.state_dict(), fname)
    
    return model

def train_adversarial_batch(train_loader, model, opt, criterion, batch_size=100, data_dir="", epochs=10, attack='fgsm', epsilon=0.01, alpha=0.01, attack_iters=40, lr_max=5e-3, lr_type="cyclic", fname="", seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

#     mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
#     train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
#     train_loader = torch.utils.data.DataLoader(zip(train_x, train_y), batch_size=batch_size, shuffle=True)

    model.train()
# 
#     opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    
    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')

#     criterion = nn.CrossEntropyLoss()

#     logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        for i, pnt in enumerate(train_loader):
            
            if isinstance(pnt, list): # For images
                X = pnt[0].view(pnt[0].shape[0], -1)
                y = pnt[1]
            else:
                X = pnt[:,0:pnt.shape[1]-1]
                y = pnt[:,pnt.shape[1]-1].long()

            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if attack == 'fgsm':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to("cpu")
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                delta = delta.detach()
            elif attack == 'none':
                delta = torch.zeros_like(X)
            elif attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
                
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
        if epoch % 1 == 0:
            print("Training epoch", epoch, "/", epochs, "loss:", loss.item(), train_acc)

#         logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
#             epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
    torch.save(model.state_dict(), fname)
    
    return model

def train_adversarial_x(dataset, model, optimizer, criterion, train_method='natural', epochs=10, batch_size=10, train_adversarial_percentage=1):
    ###### TRAIN ######
    model.train()
    best_model = model
    
    print("Start training")
    best_loss = 100.0
    
    traindata, valdata = sklearn.model_selection.train_test_split(dataset, test_size=0.2, shuffle=True, stratify=dataset.targets)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
    
        original_loss = torch.zeros((len(train_loader)))
        for batch_idx, pnt in enumerate(train_loader):
    #         print("batch", batch_idx)
            if isinstance(pnt, list): # For images
                X = pnt[0].view(pnt[0].shape[0], -1)
                y = pnt[1]
            else:
                X = pnt[:,0:pnt.shape[1]-1]
                y = pnt[:,pnt.shape[1]-1].long()
                    
            if random.uniform(0,1) < train_adversarial_percentage:
                X = create_adversarial_example(X, y, model, criterion, method=train_method, dataset=X)
            
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y).mean()
            loss.backward()
            optimizer.step()
            criterion.set_net(model)
            
            original_loss[batch_idx] = loss.data
        
        # UPDATE LEARNING RATE
        if epoch % 100 == 0:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 1
                param_group['lr'] = new_lr
            
        # VALIDATION SET        
        val_loss = torch.Tensor([])
        val_loss_at = torch.Tensor([])
        for batch_idx, pnt in enumerate(val_loader):
            if isinstance(pnt, list): # For images
                X = pnt[0].view(pnt[0].shape[0], -1)
                y = pnt[1]
            else:
                X = pnt[:,0:pnt.shape[1]-1]
                y = pnt[:,pnt.shape[1]-1].long()
            
            out = model(X)
            vloss = criterion(out, y)
            val_loss = torch.cat((val_loss,vloss))
            
            X_at = create_adversarial_example(X, y, model, criterion, method=train_method, dataset=X)
            out_at = model(X_at)
            vloss_at = criterion(out_at, y)
            val_loss_at = torch.cat((val_loss_at,vloss_at))
        
        total_val_loss = val_loss.mean()
        total_val_loss_at = val_loss_at.mean()
        if epoch % 100 == 0:
            print('Epoch', epoch, 'Loss', torch.mean(original_loss), "ValLoss", total_val_loss.data, "AttackLoss", total_val_loss_at.data)
        
        if total_val_loss.data < best_loss:
            best_loss = total_val_loss.data
            torch.save(model, "../nets/epochs_net/e{}_{}.pt".format(epoch, str(total_val_loss.data.item())[2:6]), pickle_module=dill)
            best_model = model    
    return best_model
