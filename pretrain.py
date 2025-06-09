import torch
import pyepo
from torch import nn
import numpy as np
from tqdm import tqdm
from dot.utils_data import output_paths
import dot.Environments.Warcraft as Warcraft
from dot.models_warcraft import partialResNet, shortestPathModel, deeperResNet, SmallCNN, MLP, PartialResNet34, MobileNetV2Trunk, DeeperResNet34
import argparse

parser = argparse.ArgumentParser(description='Warcraft Train on Source')
parser.add_argument('--shift', metavar='shift', type=str, choices=['orig','unif','normal'],help='Type of distribution shift')
parser.add_argument('--objective', metavar='objective', choices=['cost','cost_length','varcost_length'],type=str, help='Type of objective function',default='cost')
parser.add_argument('--loss', metavar='loss', type=str, help='Loss type', default='dfl')
parser.add_argument('--ntrain', metavar='ntrain', type=int, help='No. of training instances', default=1000)
parser.add_argument('--seed', metavar='seed', type=int, help='seed')
parser.add_argument('--seed_train', metavar='seed_train', type=int, help='seed')
parser.add_argument('--architecture', metavar='architecture', type=str, choices=['partial-resnet','partial-resnet34', 'deeper-resnet', 'deeper-resnet34', 'small-cnn', 'mlp', 'mobilenet'], help='Which predictor architecture to use.')
args = parser.parse_args()

shift=args.shift
objective = args.objective
ntrain=args.ntrain
loss_type=args.loss
seed=args.seed
seed_train=args.seed_train
architecture=args.architecture

print(f'Pretraining {architecture} on source_seed:{seed}')

k=12
path = './data'
val_frac = 0.2
epochs = 60 # number of epochs
lr = 5e-4 # learning rate
log_step = 1 # log step
batch_size = 70

#create dfl problem
problem = Warcraft.Warcraft(ntrain, ntrain, val_frac, seed, seed, shift=shift, objective=objective)

# Load dataset
X, Y, _ = problem.get_train_data()
X_val, Y_val, _ = problem.get_val_data()

# Get decisions
Z = np.array([problem.get_decision(i) for i in Y]).reshape(-1,k,k)
Z_val = np.array([problem.get_decision(i) for i in Y_val]).reshape(-1,k,k)


# TRAIN MODELS
# init model
grid = (k, k)
optmodel = shortestPathModel(grid)
if loss_type=='dfl':
    spoploss = pyepo.func.SPOPlus(optmodel, processes=1) # set loss
elif loss_type=='mse':
    spoploss = nn.MSELoss()


# Train model on Source
dataset_train = Warcraft.mapDataset(X, Y, Z)
loader_train = Warcraft.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_val = Warcraft.mapDataset(X_val, Y_val, Z_val)
loader_val = Warcraft.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

#set training seeds
torch.manual_seed(seed_train)
torch.cuda.manual_seed_all(seed_train)

if architecture == 'partial-resnet':
    nnet = partialResNet(k=k)  # init net
elif architecture == 'partial-resnet34':
    nnet = PartialResNet34(k=k)
elif architecture == 'deeper-resnet':
    nnet = deeperResNet(k=k)  # deeper version using layer1 + layer2
elif architecture == 'deeper-resnet34':
    nnet = DeeperResNet34(k=k)  # deeper version using layer1 + layer2
elif architecture == 'small-cnn':
    nnet = SmallCNN(k=k)  # simpler conv net
elif architecture == 'mlp':
    nnet = MLP(k=k)  # flat MLP on downsampled image
elif architecture=='mobilenet':
    nnet = MobileNetV2Trunk(k=k)
else:
    raise ValueError(f"Unknown architecture: {architecture}")

if torch.cuda.is_available():
    nnet = nnet.cuda()
optimizer = torch.optim.Adam(nnet.parameters(), lr=lr) # set optimizer

# train
loss_log, regret_log = [], [pyepo.metric.regret(nnet, optmodel, loader_val)]
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
    'loss_log': loss_log,
    'regret_log': regret_log
}

# Save model
file_finetuned_model, file_pretrained_model, file_regret, file_components, file_distances = output_paths('Warcraft')
output_file = file_pretrained_model(shift, ntrain, seed, seed_train, objective, None, architecture)
torch.save(nnet.state_dict(), output_file)