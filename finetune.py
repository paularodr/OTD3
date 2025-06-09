import os
import torch
import pyepo
import pickle
from torch import nn
import numpy as np
from tqdm import tqdm
from dot.utils_data import output_paths
import dot.Environments.Warcraft as Warcraft
from dot.models_warcraft import shortestPathModel, build_model
from dot.performance import evaluate_performance, predict_warcraft
from dot.losses import MSE
import argparse

parser = argparse.ArgumentParser(description='Warcraft Train on Source')
parser.add_argument('--shift', metavar='shift', type=str, choices=['orig','unif','normal'],help='Type of distribution shift')
parser.add_argument('--objective', metavar='objective', choices=['cost','cost_length','varcost_length'],type=str, help='Type of objective function',default='cost')
parser.add_argument('--loss', metavar='loss', type=str, help='Loss type', default='dfl')
parser.add_argument('--ntarget', metavar='ntarget', type=int, help='No. of training instances for fine tuning', default=1000)
parser.add_argument('--nsource', metavar='nsource', type=int, help='No. of training instances on pretraining', default=1000)
parser.add_argument('--seed', metavar='seed', type=int, help='seed')
parser.add_argument('--seed_target', metavar='seed_target', type=int, help='seed target sample')
parser.add_argument('--seed_train', metavar='seed_train', type=int, help='seed')
parser.add_argument('--architecture', metavar='architecture', type=str, choices=['partial-resnet', 'partial-resnet34', 'deeper-resnet','deeper-resnet34', 'small-cnn', 'mlp', 'mobilenet'], help='Which predictor architecture to use.')
args = parser.parse_args()

shift=args.shift
objective = args.objective
ntarget=args.ntarget
nsource=args.nsource
loss_type=args.loss
seed=args.seed
seed_target=args.seed_target
seed_train=args.seed_train
architecture=args.architecture

print(f'Finetune {architecture} pretrained with source_seed:{seed}')

k=12
path = './data'
val_frac = 0.2
freeze=False
train_stage='finetuned'
ntest=1000
domain='Warcraft'

#name files
file_finetuned_model, file_pretrained_model, file_regret, file_components, file_distances = output_paths('Warcraft')
folders_path = './models/Warcraft/finetune'
pretrained_path = file_pretrained_model(shift, nsource, seed, seed_train, objective, ntarget,architecture)
output_file = file_finetuned_model(shift, nsource, seed, seed_target, seed_train, objective, ntarget,architecture)
output_regret = file_regret(objective, shift, seed, seed_target, seed_train, train_stage, nsource, ntarget, ntest, architecture)

if not os.path.exists(output_file):
    print(f"File {output_file} does not exist. Submitting job...")

    #create dfl problem on target dataset
    problem = Warcraft.Warcraft(ntarget, ntarget, val_frac, seed_target, seed_target, shift='orig', objective=objective)

    # Load dataset
    X, Y, _ = problem.get_train_data()
    X_val, Y_val, _ = problem.get_val_data()

    # Get decisions
    Z = np.array([problem.get_decision(i) for i in Y]).reshape(-1,k,k)
    Z_val = np.array([problem.get_decision(i) for i in Y_val]).reshape(-1,k,k)


    # set parameters
    grid = (k, k)
    optmodel = shortestPathModel(grid)
    epochs = 60 # number of epochs
    lr = 5e-4 # learning rate
    log_step = 1 # log step
    batch_size = 70
    if loss_type=='dfl':
        spoploss = pyepo.func.SPOPlus(optmodel, processes=1) # set loss
    elif loss_type=='mse':
        spoploss = nn.MSELoss()


    # get train and test datasets
    dataset_train = Warcraft.mapDataset(X, Y, Z)
    loader_train = Warcraft.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Warcraft.mapDataset(X_val, Y_val, Z_val)
    loader_val = Warcraft.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # define model and set seed
    torch.manual_seed(seed_train)
    # nnet = partialResNet(k=k) # init net
    nnet = build_model(architecture,k)

    # load pretrained model
    nnet.load_state_dict(torch.load(pretrained_path))
    if torch.cuda.is_available():
        nnet = nnet.cuda()

    # Freeze all but the final layer
    if freeze:
        for param in nnet.parameters():
            param.requires_grad = False  
        for param in nnet.conv2.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, nnet.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(nnet.parameters(), lr=lr)

    # train
    mse_log, loss_log, regret_log = [], [], [pyepo.metric.regret(nnet, optmodel, loader_val)]
    tbar = tqdm(range(epochs))
    for epoch in tbar:
        nnet.train()
        for x, c, w, z in loader_train:
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = nnet(x) # predicted cost
            if loss_type=='dfl':
                loss = spoploss(cp, c, w, z).mean() # loss
            elif loss_type=='mse':
                loss = spoploss(cp, c)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log loss
            mse_log.append(MSE(cp, c))
            loss_log.append(loss.item())
            tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
        # scheduled learning rate
        if (epoch == int(epochs*0.6)) or (epoch == int(epochs*0.8)):
            for g in optimizer.param_groups:
                g['lr'] /= 10
        if epoch % log_step == 0:
            # log regret
            regret = pyepo.metric.regret(nnet, optmodel, loader_val) # regret on val
            regret_log.append(regret)

    results = {
        'mse_log': mse_log,
        'loss_log': loss_log,
        'regret_log': regret_log
    }

    # Save model
    # torch.save(nnet.state_dict(), output_file)
    # Save model results dict
else:
    print(f"File {output_file} already exists. Skipping job...")


# evaluate model performance
# get target dataset
problem_target = Warcraft.Warcraft(ntarget, ntest, val_frac, 0, 0, shift='orig', objective=objective)

#compute decision quality regret
dqr = evaluate_performance(domain, nnet, problem_target, True, True)

# compute mse
X, y_true, Y_aux = problem_target.get_test_data()
Z = np.zeros(y_true.shape)
y_pred = predict_warcraft(nnet, X, y_true, Z)
mse = ((y_true-y_pred)**2).mean()
results['mse_test'] = mse

# save regret as npy
regret=np.array(dqr)
results['regret_test']=regret
np.save(output_regret,regret)

with open(f'{output_file}.pkl', 'wb') as handle:
    pickle.dump(results, handle)
