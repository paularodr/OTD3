# OTD<sup>3</sup>: Optimal Transport Decision-Aware Dataset Distance

Code accompanying the paper ["What is the Right Notion of Distance between Predict-then-Optimize Tasks?"](https://arxiv.org/pdf/2409.06997) by Paula Rodriguez-Diaz, Lingkai Kong, Kai Wang, David Alvarez-Melis, and Milind Tambe

## Set up environment

To run the code, first you have to set up a conda environment. Once you have [Anaconda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), run the following command:
```
conda env create --name lodlenv --file=environment.yml
```
Once the environment has been created, load it using the command
```
conda activate otd3env
```
## Usage

The experimental results in the paper consist of two main steps:  
1. Computing dataset distances  
2. Training and evaluating models using a decision-focused loss  

After obtaining both the dataset distance and the performance of the fine-tuned model, our main analysis focuses on the correlation between these two metrics.
Below, we provide an example demonstrating how to implement steps 1 and 2 in the Warcraft setting.

### 1. Compute dataset distance
```
python3 components.py --domain Warcraft --ntarget 1000 --seed_target 0 --nsource 100 --objective cost --shift unif
python3 distances.py  --domain Warcraft --ntarget 1000 --seed_target 0 --nsource 100 --objective cost --shift unif
```

### 2. Train and evaluate PtO performance
```
python3 pretrain.py --problem warcraft --objective cost --ntrain 100 --seed 0 --shift unif --architecture deeper-resnet
python3 finetune.py --problem warcraft --objective cost --seed 0 --seed_target 1 --ntarget 1000 --nsource 100 --shift unif --objective cost --architecture deeper-resnet
```

## Acknowledgments  
This repository makes use of several external packages:  
- The [POT](https://pythonot.github.io/) package for computing dataset distances using the internal EMD and Sinkhorn algorithms.  
- The [LODLs](https://github.com/sanketkshah/LODLs/tree/main) and [PyEPO](https://github.com/khalil-research/PyEPO) packages for implementing and training predict-then-optimize benchmark models.
