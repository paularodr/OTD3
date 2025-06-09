import random
import torch
import time
import numpy as np
import pandas as pd
from dot.models_warcraft import shortestPathModel

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# fix random seed
random.seed(135)
np.random.seed(135)
torch.manual_seed(135)

def read_warcraft(k, path, type='train'):
    folder = f"{path}/warcraft_shortest_path_oneskin/{k}x{k}"
    tmaps = np.load(f"{folder}/{type}_maps.npy")
    costs = np.load(f"{folder}/{type}_vertex_weights.npy")
    paths = np.load(f"{folder}/{type}_shortest_paths.npy")
    return tmaps, costs, paths

class mapDataset(Dataset):
    def __init__(self, tmaps, costs, paths):
        self.tmaps = tmaps
        self.costs = costs
        self.paths = paths
        self.objs = (costs * paths).sum(axis=(1,2)).reshape(-1,1)

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, ind):
        return (
            torch.FloatTensor(self.tmaps[ind].transpose(2, 0, 1)/255).detach(), # image
            torch.FloatTensor(self.costs[ind]).reshape(-1),
            torch.FloatTensor(self.paths[ind]).reshape(-1),
            torch.FloatTensor(self.objs[ind]),
        )

def create_dataset(tmaps, costs, paths):
    tmaps_train, costs_train, paths_train = tmaps[0], costs[0], paths[0]
    tmaps_val, costs_val, paths_val = tmaps[1], costs[1], paths[1]
    tmaps_test, costs_test, paths_test = tmaps[2], costs[2], paths[2]

    dataset_train = mapDataset(tmaps_train, costs_train, paths_train)
    dataset_val = mapDataset(tmaps_val, costs_val, paths_val)
    dataset_test = mapDataset(tmaps_test, costs_test, paths_test)

    return dataset_train, dataset_val, dataset_test

def create_dataloader(dataset_train, dataset_val, dataset_test, batch_size):
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val, loader_test


class Warcraft():
    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=100,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
        label_seed=0, #seed for unif sampeled costs
        shift='orig', # dataset type for output,
        objective='cost', #type of objective function 
        std=0.5, #frac for standar deviation for normal distribution shift
        k=12,
        data_path='./data'
    ):
        super(Warcraft, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        # Generate Dataset
        #   Save relevant parameters
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.shift = shift
        self.k = k
        self.data_path = data_path
        self.label_seed = label_seed
        self.std = std
        self.objective = objective
        self.labels=[0.8, 1.2, 5.3, 7.7, 9.2]

        #opt model
        grid = (self.k,self.k)
        self.opt = shortestPathModel(grid)
        self.model_name = 'resnet'

        # random seed setting
        self._set_seed()

        #   Generate datasets
        X_train, Y_train, Z_train = read_warcraft(self.k, data_path, type='train')
        X_test, Y_test, Z_test = read_warcraft(self.k, data_path, type='test')
        X_val, Y_val, Z_val = read_warcraft(self.k, data_path, type='val')

        # create label shift
        orig_costs = np.array(self.labels)
        if self.shift == 'orig':
            assert 0 < val_frac < 1
            self.val_frac = val_frac
            self.val_idxs = range(0, int(self.val_frac * num_train_instances))
            self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
            assert all(x is not None for x in [self.train_idxs, self.val_idxs])
            self.Xs_train, self.Ys_train = X_test[self.train_idxs], Y_test[self.train_idxs]
            self.Xs_val, self.Ys_val = X_val[self.val_idxs], Y_val[self.val_idxs]
            self.Xs_test, self.Ys_test = X_test, Y_test
        else:
            if self.shift == 'unif':
                np.random.seed(self.label_seed)
                new_costs = np.random.uniform(orig_costs.min(),orig_costs.max(),orig_costs.shape[0])
            elif self.shift == 'normal':
                np.random.seed(self.label_seed)
                new_costs = np.array([np.random.normal(loc=x, scale=x*self.std) for x in orig_costs])
            self.labels = new_costs
            Y_train_orig, Y_test_orig, Y_val_orig = Y_train.copy(), Y_test.copy(), Y_val.copy()
            for i,y in enumerate(orig_costs):
                Y_train[Y_train_orig==y] = new_costs[i]
                Y_test[Y_test_orig==y] = new_costs[i]
                Y_val[Y_val_orig==y] = new_costs[i]


            self.train_idxs = np.random.choice(X_train.shape[0], int((self.num_train_instances)*(1-val_frac)), replace=False)
            self.val_idxs = np.random.choice(X_val.shape[0], int((self.num_train_instances)*(val_frac)), replace=False)
            self.test_idxs = np.random.choice(X_test.shape[0], self.num_test_instances, replace=False)

            self.Xs_train, self.Ys_train = X_train[self.train_idxs], Y_train[self.train_idxs]
            self.Xs_test, self.Ys_test = X_test[self.test_idxs], Y_test[self.test_idxs]
            self.Xs_val, self.Ys_val = X_val[self.val_idxs], Y_val[self.val_idxs]
                
    def _set_seed(self, rand_seed=int(time.time())):
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)

    def get_train_data(self):
        return self.Xs_train, self.Ys_train, [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_val, self.Ys_val, [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test, [None for _ in range(len(self.Ys_test))]

    def get_dfl_dataset(self, **kwargs):
        X_train, Y_train, Y_train_aux = self.get_train_data()
        X_val, Y_val, Y_val_aux = self.get_val_data()
        X, Y = np.concatenate((X_train,X_val)), np.concatenate((Y_train,Y_val))
        Y_aux = [None]*Y.shape[0]
        Z = self.get_decision(Y, aux_data=Y_aux, isTrain=True) #decisions
        obj = self.get_objective(Y, Z, aux_data=Y_aux) #objetive value
        return X, Y, Y_aux, Z, obj

    def get_modelio_shape(self):
        return self.Xs.shape[-1], 1

    def get_twostageloss(self):
        return 'mse'

    def get_objective(self, Y, Z, **kwargs):
        Z = Z.reshape(-1,self.k,self.k)
        Y = Y.reshape(-1,self.k,self.k)
        if self.objective == 'cost':
            obj = (Z * Y).sum(axis=1).sum(axis=1)
        elif self.objective == 'cost_length': 
            obj = (Z * Y).sum(axis=1).sum(axis=1) + Z.sum(axis=1).sum(axis=1)
        elif self.objective == 'varcost_length':
            obj = ((Z.std(axis=1))**2).sum(axis=1).sum(axis=1) + Z.sum(axis=1).sum(axis=1)
        return obj

    def get_decision(self, Y, **kwargs):
        if len(Y.shape)==3:
            decision = []
            for y in Y:
                self.opt.setObj(y,self.objective)
                sol,_ = self.opt.solve()
                decision.append(sol)
            decision = np.array(decision).reshape(-1,self.k,self.k)
        else:
            self.opt.setObj(Y,self.objective)
            decision,_ = self.opt.solve()
        return decision

    def get_output_activation(self):
        return None
    