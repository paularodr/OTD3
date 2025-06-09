import ot
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from dot.utils_data import output_paths
from dot.utils_distance import otdistance

# arguments
parser = argparse.ArgumentParser(description='Compute distance components')
parser.add_argument('--domain', metavar='domain', type=str, choices=['CubicTopK', 'Warcraft'], help='domain for experiment')
parser.add_argument('--shift', metavar='shift', type=str, choices=['orig','unif','normal'],help='Type of distribution shift')
parser.add_argument('--ntarget', metavar='ntarget', type=int, help='Size of target dataset')
parser.add_argument('--seed_target', metavar='seed_target', type=int, help='seed target sample')
parser.add_argument('--nsource', metavar='nsource', type=int, help='Size of source dataset')
parser.add_argument('--objective', metavar='objective', choices=['cost','cost_length','varcost_length'],type=str, help='Type of objective function',default='cost')
args = parser.parse_args()

domain=args.domain
shift=args.shift
ntarget=args.ntarget
seed_target=args.seed_target
nsource=args.nsource
objective=args.objective

# additional arguments (fixed)
data_dir = f'./data/{domain}'
normalize_by = 'mean'
dflalpha=0.001

# file paths
_,_,_, file_components, file_distances = output_paths(domain,dflalpha=dflalpha)
components_file = file_components(shift,objective,ntarget,nsource,seed_target)
output_file = file_distances(shift, objective, ntarget, nsource,seed_target)

# weight for components
samples_per_dim = 5
values = np.arange(samples_per_dim)
alphas = np.array([[i[0]/sum(i),i[1]/sum(i),i[2]/sum(i)] for i in product(values, repeat = 3) if sum(i)!=0])
alphas = np.unique(alphas,axis=0)


# Load ground cost components
with open(components_file, 'rb') as f:
    components = pickle.load(f)

C_x_target, C_y_target, C_w_target = components['target']['x'], components['target']['y'], components['target']['w']
if normalize_by == 'mean':
    M_x, M_y, M_w = C_x_target.mean(), C_y_target.mean(), C_w_target.mean()
elif normalize_by == 'max':
    M_x, M_y, M_w = C_x_target.max(), C_y_target.max(), C_w_target.max()


emd_distances = {}
sinkhorn_distances = {}
for seed in tqdm(range(30)):
    key = f'seed_{seed}'
    C_x, C_y, C_w = components[key]['x'], components[key]['y'], components[key]['w']
    C_x, C_y, C_w = C_x/M_x, C_y/M_y, C_w/M_w

    emd_dist = []
    sinkhorn_dist = []
    for a in alphas:
        C = a[0]*C_x + a[1]*C_y + a[2]*C_w 
        emd = otdistance(C)
        snkh = otdistance(C, sinkhorn=True, reg=0.1)
        emd_dist.append(emd)
        sinkhorn_dist.append(snkh)

    emd_distances[key] = np.array(emd_dist)
    sinkhorn_distances[key] = np.array(sinkhorn_dist)


with open(f'{output_file}_emd.pkl', 'wb') as f:
        pickle.dump(emd_distances, f)

with open(f'{output_file}_sinkhorn.pkl', 'wb') as f:
        pickle.dump(sinkhorn_distances, f)

print(f'Successfully saved distances: {output_file}')