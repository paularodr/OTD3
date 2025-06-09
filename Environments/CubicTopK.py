'''
Adapted from from Shah et. al. 2022
https://github.com/sanketkshah/LODLs
'''

from functools import partial
from unicodedata import decimal

import torch
from torch.distributions import Normal, Bernoulli
import numpy as np
import random
import pdb
import time

from .solvers import TopK_custom

def poly(x, coef=0.65):
    y = 10*(((x)**3) - coef*x)
    return y.squeeze()

def eval_poly(x, dataset='C'):
    if dataset=='C':
        y = 10*((x**3) - 0.65*x)
    elif dataset=='A':
        y = 10*(((x)**3) - 0.85*x)*((x+1))
    elif dataset=='B':
        y = 10*(((x)**3) - 0.85*x)*(((-1)*x+1))
    return y.squeeze()

def draw_samples(num_instances, num_items, lb=-1, up=1, dataset='C'):
    X = (up-lb) * torch.rand(num_instances, num_items, 1) + lb
    Y = eval_poly(X, dataset)
    Y = Y.reshape(num_instances,num_items)
    return X,Y

class CubicTopK():
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=100,  # number of instances to use from the dataset to test
        num_items=50,  # number of targets to consider
        budget=2,  # number of items that can be picked
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        coef=0.65, # dataset type for output
    ):
        super(CubicTopK, self).__init__()
        self.model_name = 'dense'
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        self._set_seed(rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)

        #set device
        self.device=self._set_device()

        # Generate Dataset
        #   Save relevant parameters
        self.num_items = num_items
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.coef = coef
        #   Generate features
        self._set_seed(train_seed)
        self.Xs_train = 2 * torch.rand(self.num_train_instances, self.num_items, 1) - 1
        self._set_seed(test_seed)
        self.Xs_test = 2 * torch.rand(self.num_test_instances, self.num_items, 1) - 1
        #   Generate Labels
        self.Ys_train = poly(self.Xs_train, coef=self.coef)
        self.Ys_test = poly(self.Xs_test, coef=self.coef)
        
        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Save variables for optimisation
        assert budget < num_items
        self.budget = budget

        # Undo random seed setting
        self._set_seed()

    def _set_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def _set_seed(self, rand_seed=int(time.time())):
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs], [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs], [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test, [None for _ in range(len(self.Ys_test))]

    def get_dfl_dataset(self, **kwargs):
        X_train, Y_train, Y_train_aux = self.get_train_data()
        X_val, Y_val, Y_val_aux = self.get_val_data()
        X_train, Y_train = torch.tensor(X_train), torch.tensor(Y_train)
        X_val, Y_val = torch.tensor(X_val), torch.tensor(Y_val)
        X, Y = torch.cat([X_train,X_val]), torch.cat([Y_train,Y_val])
        Z = self.get_binary_decision(X,Y) #decisions
        Y_sorted = self.sort_instance(X,Y)
        obj = self.get_objective(Y_sorted, Z) #objetive value
        return X, Y, None, Z, obj

    def get_objective(self, Y, Z, **kwargs):
        return (Z * Y).sum(dim=-1)

    def opt_train(self, Y):
        gamma = TopK_custom(self.budget)(-Y).squeeze()
        Z = gamma[...,0] * Y.shape[-1]
        return Z
    
    def opt_test(self, Y):
        _, idxs = torch.topk(torch.tensor(Y), self.budget)
        Z = torch.nn.functional.one_hot(idxs, Y.shape[-1])
        return Z if self.budget == 0 else Z.sum(dim=-2)

    def get_decision(self, Y, isTrain=False, **kwargs):
        return self.opt_train(Y) if isTrain else self.opt_test(Y)

    def sort_instance(self,X,Y):
        X=torch.tensor(X)
        args = torch.sort(X,dim=1)[1].squeeze()
        Y_sorted = Y[torch.arange(Y.shape[0]).unsqueeze(1), args]
        return Y_sorted
    
    def get_binary_decision(self,X,Y):
        Y_sorted = self.sort_instance(X,Y)
        Z = np.zeros(Y.shape)
        for i in Y_sorted.argmax(dim=1):
            Z[:,i]=1
        Z = torch.tensor(Z)
        return Z
    
    def get_modelio_shape(self):
        return 1, 1

    def get_output_activation(self):
        return None
    
    def get_twostageloss(self):
        return 'mse'


# Unit test for RandomTopK
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pdb

    # Load An Example Instance
    pdb.set_trace()
    problem = CubicTopK()

    # Plot It
    Xs = problem.Xs_train.flatten().tolist()
    Ys = problem.Ys_train.flatten().tolist()
    plt.scatter(Xs, Ys)
    plt.show()