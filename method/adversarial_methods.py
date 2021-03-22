import torch
import numpy as np
import random
from method.utils import or_float_tensors
import torchattacks
from scipy.spatial import distance
from method import cwbase

# Create adversarial example using a certain method
def create_adversarial_example(x, y, model, loss_fct, iterations=200, method='natural', mal_index=0, dataset={}, report_loss_diff=False, loss_fct_adv=None, epsilon=0.02, optimizer=None):
   
    if method == 'natural':
        return x
    elif method == 'genneuact':
        return genneuact(x, y, model, loss_fct, is_report_loss_diff=True, dataset=dataset)
    elif method == 'ta_fgsm':
        return ta_fgsm(x, y, model)
    elif method == 'ta_deepfool':
        return ta_deepfool(x, y, model)
    elif method == 'ta_bim':
        return ta_bim(x, y, model)
    elif method == 'ta_cw':
        return ta_cw(x, y, model)
    elif method == 'ta_rfgsm':
        return ta_rfgsm(x, y, model)
    elif method == 'ta_pgd':
        return ta_pgd(x, y, model)
    
    else:
        raise Exception('No such inner maximizer algorithm')

# Create adversarial examples using the C&W implementation from Kaiwen Wu 
def cw(x, y, model):
    adversary = cwbase.L2Adversary(targeted=False, box=(0.,1.))
    advs = adversary(model, x, y, to_numpy=False)
    
    return advs

# Create adversarial examples using the FGSM implementation from Torchattacks
def ta_fgsm(x, y, model, epsilon=0.007):
    attack = torchattacks.FGSM(model, eps=epsilon)
    advs = attack(x, y)
    return advs

# Create adversarial examples using the DeepFool implementation from Torchattacks
def ta_deepfool(x, y, model, steps=3):
    attack = torchattacks.DeepFool(model, steps=steps)
    advs = attack(x, y)
    return advs

# Create adversarial examples using the BIM implementation from Torchattacks
def ta_bim(x, y, model, eps=4/255, alpha=1/255, steps=0):
    attack = torchattacks.BIM(model, eps= eps, alpha=alpha, steps=steps)
    advs = attack(x, y)
    return advs

# Create adversarial examples using the C&W implementation from Torchattacks
def ta_cw(x, y, model, c=0.001, kappa=0, steps=1000, lr=0.01):
    print(c)
    attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
    advs = attack(x, y)
    return advs

# Create adversarial examples using the RFGSM implementation from Torchattacks
def ta_rfgsm(x, y, model, eps=16/255, alpha=8/255, steps=1):
    attack = torchattacks.RFGSM(model, eps=eps, alpha=alpha, steps=steps)
    advs = attack(x, y)
    return advs

# Create adversarial examples using the PGD implementation from Torchattacks
def ta_pgd(x, y, model, eps=0.3, alpha=2/255, steps=40, random_start=False):
    attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
    advs = attack(x, y)
    return advs

'''
GenNeuAct: a new method for generating adversarial examples. The basis of the method is a genetic 
algorithm, with a fitness function based on neuron activation values. The fitness of a data point 
is determined by scoring it against the estimated density function of each neuron and each output 
class.
'''
def genneuact(x, y, model, criterion, epsilon=0.02, alpha=0.5, is_report_loss_diff=False, use_sample=False, dataset={}):
    # Change x to 2 dimensional data
    x = x.view(x.shape[0], -1)
    
    # Settings genetic algorithm
    num_pts = x.shape[0]   # batch size
    pnt_size = x.shape[1]   # length data point
    pop_size = 6
    num_gen = 100
    num_par = 2
    scale = 0.0075
    scale_mut = 0.05 # Digits: 0.035, cancer: 0.005, fashion: 0.09, diabetes: 0.05
    mut_rate = 0.1
    
    clip_min = 0
    clip_max = 1
    
    # Initialize population
    pop = torch.zeros((num_pts, pop_size, pnt_size))
    new_y = y.repeat_interleave(pop_size)
    
    # Initialize final result to be returned
    res = torch.zeros_like(x) - 1

    # Generate initial population   
    for mal_pnt in range(num_pts): # for every data point in x (batch-size)
        pop[mal_pnt][0] = x[mal_pnt, :]
        # Pick pop_pts - 1 points from the benign dataset, and add a small perturbation to it to create initial population
        for i in range(pop_size - 1):
            delta = torch.zeros_like(x[mal_pnt])
            delta = torch.Tensor(np.random.normal(scale=scale, size=x[mal_pnt].shape))
            pop[mal_pnt][i + 1] = torch.clamp(x[mal_pnt] + delta, clip_min, clip_max)

    # Run generations
    for gen in range(num_gen):
        # Make sure we are staying between the clipping values
        pop = np.clip(pop, clip_min, clip_max)
        
        # Compute fitness of current population
        fitness = criterion.neuron_fitness(pop, new_y).numpy()

        # Selecting the best parents in the population for mating.
        parents = torch.zeros((num_pts, num_par, pnt_size))
        fitness = np.reshape(fitness, (num_pts, pop_size))
        
        for pnt in range(num_pts):
            for parent_num in range(num_par):
                max_fitness_idx = np.where(fitness[pnt] == np.max(fitness[pnt]))
                max_fitness_idx = max_fitness_idx[0][0]
                parents[pnt, parent_num] = pop[pnt, max_fitness_idx]
                fitness[pnt, max_fitness_idx] = -999

        # Generating next generation using crossover.
        offspring_size = (num_pts, pop_size - num_par, pnt_size)
        offspring = np.empty(offspring_size)

        for pnt in range(num_pts):
            
            for k in range(offspring_size[1]):
                # Index of the parents to mate
                parent1_idx = k % num_par
                parent2_idx = (k+1) % num_par
                
                # Method from lit study
                a = 0.366
                
                di = parents[pnt, parent1_idx] - parents[pnt, parent2_idx]
                xmin = np.minimum(parents[pnt, parent1_idx], parents[pnt, parent2_idx]) - a * di
                xmax = np.maximum(parents[pnt, parent1_idx], parents[pnt, parent2_idx]) + a * di
                offspring[pnt,k] = np.clip(np.random.uniform(xmin,xmax, size=len(di)), clip_min, clip_max)
        
        # Mutate offspring by drawing a random sample from the Gaussian distribution, with a standard deviation determined in the settings
        num_mut = int(mut_rate * offspring.shape[0] * offspring.shape[1] * offspring.shape[2])
        for mut in range(num_mut):
            rand_add = np.random.normal(scale = scale_mut)
            rand_idx_0 = random.randint(0, offspring.shape[0] - 1)
            rand_idx_1 = random.randint(0, offspring.shape[1] - 1)
            rand_idx_2 = random.randint(0, offspring.shape[2] - 1)
            offspring[rand_idx_0, rand_idx_1, rand_idx_2] = np.clip(offspring[rand_idx_0, rand_idx_1, rand_idx_2] + rand_add,clip_min,clip_max)

        # Creating the new population based on the parents and offspring.
        pop[:, 0:parents.shape[1]] = parents
        pop[:, parents.shape[1]:, :] = torch.Tensor(offspring)
               
        
        # Check if adversarial examples were found and store them in res
        data = pop.view(num_pts * pop_size, - 1)
        predict_out = model(data)
        _, predict_y = torch.max(predict_out, 1)

        for idx, p_y in enumerate(predict_y):
            pnt_idx = int(np.floor(idx / pop_size))
            
            if not p_y == new_y[idx]:
                if res[pnt_idx][0] == -1:
                    res[pnt_idx] = data[idx]
                elif distance.euclidean(res[pnt_idx], x[pnt_idx]) > distance.euclidean(data[idx], x[pnt_idx]):
                    res[pnt_idx] = data[idx]

            
        
    # Get best solution from final population, if adversarial examples were not found yet
    fitness = criterion.neuron_fitness(pop, new_y).numpy()

    for pnt in range(num_pts):
        if res[pnt][0] == -1:
            max_fitness_idx = np.where(fitness[pnt] == np.max(fitness[pnt]))
            max_fitness_idx = max_fitness_idx[0][0]
            res[pnt] = pop[pnt, max_fitness_idx]
        
    return res

def round_x(x, alpha=0.5):
    """
    rounds x by thresholding it according to alpha which can be a scalar or vector
    :param x:
    :param alpha: threshold parameter
    :return: a float tensor of 0s and 1s.
    """
    return (x > alpha).float()

def get_x0(x, is_sample=False):
    """
    Helper function to randomly initialize the the inner maximizer algos
    randomize such that the functionality is preserved.
    Functionality is preserved by maintaining the features present in x
    :param x: training sample
    :param is_sample: flag to sample randomly from feasible area or return just x
    :return: randomly sampled feasible version of x
    """
    if is_sample:
        rand_x = round_x(torch.rand(x.size()))
        if x.is_cuda:
            rand_x = rand_x.cuda()
        return or_float_tensors(x, rand_x)
    else:
        return x


