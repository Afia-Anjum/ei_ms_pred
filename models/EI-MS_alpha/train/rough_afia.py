import torch
import torch.nn as nn
import numpy as np
import tqdm

from arguments import get_args, create_dirs
from train.train_base import train_model, test_model
from models.prop_predictor import PropPredictor
from datasets.mol_dataset import get_loader
from graph.mol_graph import MolGraph
from utils import data_utils, train_utils, write_utils
import pdb

parameters={"loss":['ls', 'lad', 'huber'],
    #"loss":['ls', 'lad', 'huber', 'quantile'],
    #"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "learning_rate": [0.05,0.1, 0.15, 0.2],
    #"min_samples_split": [2,4,6], #doesn't matter
    #"min_samples_leaf": [1,2,4,6,8], #doesn't matter
    #"max_depth":[3,5,8,10,12,15],
    "max_depth":[3,5,8,10],
    #"max_features":["log2","sqrt"],
    "max_features":["sqrt"],
    #"criterion": ["mae"],
    "subsample":[0.5,0.7,0.8, 0.85],
    #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10,100,200,300,400]
    }
	
args=parameters ##change it
raw_data = data_utils.read_smiles_from_file('%s/raw.csv' % args.data)
n_classes = 1
prop_predictor, optimizer=init_model(args, n_classes)
data_utils.load_shortest_paths(args)

agg_stats = ['loss', 'nei_score','mae','gnorm', 'gnorm_clip']
selection_stat = 'mae'

all_stats = {}
for name in agg_stats:
	all_stats[name] = []


















