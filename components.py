import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# dot package
from dot.Environments.CubicTopK import CubicTopK
from dot.Environments.Warcraft import Warcraft
from dot.models_warcraft import partialResNet
from dot.components import qreg_target_source, summary_stats, get_standard_component, get_decision_component
from dot.utils_data import output_paths

# arguments
parser = argparse.ArgumentParser(description='Compute distance components')
parser.add_argument('--domain', metavar='domain', type=str, choices=['CubicTopK', 'Warcraft'], help='domain for experiment', )
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
layers=1
dflalpha=0.001
val_frac=0.2
nseeds = 30
loss='dfl'
data_dir = f'./data/{domain}'
ntest=1000

if domain == 'Warcraft':
    k=120
    sample_seed=0
    _, _, _, file_components, file_distances = output_paths(domain)
elif domain == 'CubicTopK':
    samples=100
    budget=1
    _, _, _, file_components, file_distances = output_paths(domain, dflalpha=dflalpha)
    coefficients = [0, 0.1, 0.2, 0.3, 0.4, 0.65, 0.9, 1.0, 1.1, 1.2, 1.3]

output_file = file_components(shift,objective,ntarget,nsource,seed_target)

# get target dataset
if domain == 'Warcraft':
    problem_target = Warcraft(ntarget, ntarget, val_frac, seed_target, seed_target, shift='orig', objective=objective)
elif domain == 'CubicTopK':
    problem_target = CubicTopK(ntarget, ntarget, samples, budget, val_frac, 5, 0.65)
Xt, Yt, Yt_aux, Zt, obj_t = problem_target.get_dfl_dataset()

# Xt=Xt.reshape(ntarget,samples)

# get ground cost component from target to target
Cx_t = get_standard_component(Xt, Xt)
Cy_t = get_standard_component(Yt, Yt)
Cw_t = get_decision_component(problem_target, problem_target)
components = {'target':{'x':Cx_t, 'y':Cy_t, 'w':Cw_t}}

# get ground cost component from target to source datasets
for seed in tqdm(range(30)):
    problem_source = Warcraft(nsource, nsource, val_frac, seed, seed, shift=shift,objective=objective)
    Xs, Ys, Ys_aux, Zs, obj_s = problem_source.get_dfl_dataset()
    Cx = get_standard_component(Xs, Xt)
    Cy = get_standard_component(Ys, Yt)
    Cw = get_decision_component(problem_source, problem_target)
    components[f'seed_{seed}'] = {'x':Cx, 'y':Cy, 'w':Cw}

with open(output_file, 'wb') as f:
    pickle.dump(components, f)

print(f'Successfully saved components: {output_file}')