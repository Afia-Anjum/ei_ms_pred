import torch
import torch.nn as nn
import numpy as np
import tqdm
from math import sqrt

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from arguments import get_args, create_dirs
from train.train_base import train_model, test_model
from models.prop_predictor import PropPredictor
from datasets.mol_dataset import get_loader
from graph.mol_graph import MolGraph
from utils import data_utils, train_utils, write_utils
import pdb

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
default_collate_func = dataloader.default_collate

def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)


for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


def init_model(args, n_classes):
    prop_predictor = PropPredictor(args, n_classes=n_classes)
    prop_predictor.to(args.device)

    optimizer = torch.optim.Adam(prop_predictor.parameters(), lr=args.lr)
    return prop_predictor, optimizer


def load_datasets(raw_data, split_idx, args):
    data_splits = data_utils.read_splits('%s/split_%d.txt' % (args.data, split_idx))
    dataset_loaders = {}
    if not args.test_mode:
        #changed_by_Afia: shuffle=True to False, Oct 13, 2021, very useful when new predictions are done. 
        #Again changed_by_Afia: shuffle=False to True, Oct 21, 2021, when a new model is trained for EI-MS prediction. changed to original: Oct 24 

        dataset_loaders['train'] = get_loader(
            raw_data, data_splits['train'], args, shuffle=True)
        #dataset_loaders['train'] = get_loader(
        #    raw_data, data_splits['train'], args, shuffle=False)
        dataset_loaders['valid'] = get_loader(
            raw_data, data_splits['valid'], args, shuffle=False)
    dataset_loaders['test'] = get_loader(
        raw_data, data_splits['test'], args, shuffle=False)
    return dataset_loaders


def get_test_loader(raw_data, split_idx, args):
    data_splits = data_utils.read_splits('%s/split_%d.txt' % (args.data, split_idx))
    test_loader = get_loader(raw_data, data_splits['test'], args,
                             shuffle=False, num_workers=0, batch_size=1)
    return test_loader


def scatter_by_anchor_indices(MW_list,pred_logits,n_data):
    max_prediction_above_molecule_mass=5
    data=pred_logits.clone()
    data_copy=pred_logits.clone()
    m = nn.ReLU()
    data=m(data)
    data_copy=m(data_copy)
    #print("Printing data before:")
    #print(data)
    
    #print(MW_list)
    
    #do I need to empty out the data and replace everything with Zero??

    
    for i in range(n_data):
        for j in range(1000):
            #print("PRINT START:")
            #print(data[i][j])
            if (MW_list[i] - 1 - j + max_prediction_above_molecule_mass) >= 0:
                data[i][j]=data_copy[i][MW_list[i] - 1 - j + max_prediction_above_molecule_mass]
            #print(data[i][j])
            #print(data_copy[i][MW_list[i] - 1 - j + max_prediction_above_molecule_mass])

    #print("Printing data after:")
    #print(data)
    #molecule_mass - j + index_shift
    #num_data_columns = data.shape
    #print(num_data_columns)
    #indices = np.arange(1000)[np.newaxis, ...]
    #print(indices)
    
    #sys.exit()
    return data

def reverse_prediction(n_data,pred_logits,smiles_list):
    MW_list=[]
    for i in range(len(smiles_list)):
        p=smiles_list[i]
        MW_list.append(int(ExactMolWt(Chem.MolFromSmiles(p))))
    data=scatter_by_anchor_indices(MW_list,pred_logits,n_data)
    return data

def mask_prediction_by_mass(n_data,pred_logits,smiles_list):
    #print(list(pred_logits.size())[0])
    #print(ExactMolWt(Chem.MolFromSmiles('CCOC(=O)COC(=O)C(F)(F)Cl')))
    MW_list=[]
    for i in range(len(smiles_list)):
        p=smiles_list[i]
        MW_list.append(int(ExactMolWt(Chem.MolFromSmiles(p))))
    #print(MW_list)
    for i in range(n_data):
        for j in range(1000):
            if j > (MW_list[i]-1+5):
                pred_logits[i][j]=0.0
    
    m = nn.ReLU()
    pred_logits_relued=m(pred_logits)
    return pred_logits_relued, pred_logits_relued

def make_mass_intensity_pairs(true_logits,MS_labels_list,n_data):
    #true_logits=pred_logits.clone().detach()
    #true_logits[true_logits!=0]=0
    #print(pred_logits)
    for i in range(n_data):
        peak=MS_labels_list[i]
        #print(peak)
        p=peak.split(";")
        peak_locs=[]
        peak_intensities=[]
        for l in range(len(p)-1):
            loc=int(p[l].split()[0])
            intensity=float(p[l].split()[1])
            #print(loc)
            #print(intensity)
            #print(i)
            true_logits[i][loc]=intensity
        #print(true_logits[i][loc])
    return true_logits

def make_weights(true_logits):
    num_bins=true_logits.shape[1]
    weights=np.power(np.arange(1, num_bins + 1),0.5)[np.newaxis, :]
    #print("Printing weights:")
    return weights/np.sum(weights)

def make_loss(pred_logits_relued,true_logits):
    #print("Printing true:")
    #print(true_logits.size())
    #print("Printing pred_logits:")
    #print(pred_logits_relued.size())
    weights=make_weights(true_logits)
    #print("Printing weights")
    #print(weights.shape)
    a=torch.abs(pred_logits_relued - true_logits)
    a=a*a
    weights=weights.argmax()
    weighted_square_error=a*weights
    #print(weighted_square_error.size())
    t=torch.mean(weighted_square_error)
    #print(t)
    return t
    #previously working wrong version
    #weights=make_weights(true_logits)
    #print("Printing weights")
    #a=true_logits-pred_logits_relued
    #a=a*a    
    #np_arr_a = a.cpu().detach().numpy()
    #weights=np.transpose(weights)
    #loss=np.mean(weighted_square_error.reshape(-1))
    #t=torch.from_numpy(np.asarray(loss))
    #return t
        
def main():
    args = get_args()

    model_types = ['conv_net', 'conv_net_attn', 'transformer']
    assert args.model_type in model_types

    if args.multi:
        raw_data = data_utils.read_smiles_multiclass('%s/raw.csv' % args.data)
        #the output raw_data is coming from data_utils.py file where the returned values are in the format: smiles_data.append((smiles, MW_labels, MS_labels))
        #Afia: number of classes will change to 1000(max peak loc).
        #n_classes = len(raw_data[0][1])
        n_classes = 1000
    else:
        raw_data = data_utils.read_smiles_from_file('%s/raw.csv' % args.data)
        n_classes = 1

    prop_predictor, optimizer = init_model(args, n_classes)
    data_utils.load_shortest_paths(args)  # Shortest paths includes all splits

    agg_stats = ['loss', 'nei_score']
    if args.loss_type == 'ce':
        agg_stats += ['acc', 'auc']
    elif args.loss_type == 'mae':
        agg_stats += ['mae']
    #Afia: added a new type of loss specifically for our prediction results. In paper it's written that the loss is a modified mean squared error.
    elif args.loss_type == 'generalized_mse':
        agg_stats += ['generalized_mse']


    stat_names = agg_stats + ['gnorm', 'gnorm_clip']

    selection_stat = 'loss'
    select_higher = False
    if args.loss_type == 'ce':
        selection_stat = 'auc'
        select_higher = True
    if args.loss_type == 'mae':
        selection_stat = 'mae'
    if args.loss_type == 'generalized_mse':
        selection_stat = 'generalized_mse'


    if args.test_mode:
        dataset_loaders = load_datasets(raw_data, 0, args)

        test_model(
            dataset_loaders=dataset_loaders,
            model=prop_predictor,
            stat_names=stat_names,
            train_func=run_epoch,
            args=args,)

        exit()

    all_stats = {}
    for name in agg_stats:
        all_stats[name] = []
    output_dir = args.output_dir
    all_model_paths = []

    for round_idx in range(args.n_rounds):
        dataset_loaders = load_datasets(raw_data, round_idx, args)
        prop_predictor, optimizer = init_model(args, n_classes)

        cur_output_dir = '%s/run_%d' % (output_dir, round_idx)
        args.output_dir = cur_output_dir
        create_dirs(args, cur_output_dir)

        test_stats, best_model_path = train_model(
            dataset_loaders=dataset_loaders,
            model=prop_predictor,
            optimizer=optimizer,
            stat_names=stat_names,
            selection_stat=selection_stat,
            train_func=run_epoch,
            args=args,
            select_higher=select_higher,)

        # Aggregate stats of interest
        for name in agg_stats:
            all_stats[name].append(test_stats[name])
        all_model_paths.append(best_model_path)

    if args.loss_type == 'mse':
        loss_arr = all_stats['loss']
        rmse_arr = [x ** 0.5 for x in loss_arr]
        all_stats['rmse'] = rmse_arr

    # Write summary file
    summary_file = open('%s/summary.txt' % output_dir, 'w+')

    for name, stats_arr in all_stats.items():
        stats = np.array(stats_arr)
        mean, std = np.mean(stats), np.std(stats)
        stats_string = '%s: %s, mean: %.7f, std: %.7f' % (name, str(stats_arr), mean, std)
        print(stats_string)
        summary_file.write('%s\n' % stats_string)

    for model_path in all_model_paths:
        summary_file.write('%s\n' % model_path)

    summary_file.close()


def run_epoch(data_loader, model, optimizer, stat_names, args, mode,
              write_path=None):
    training = mode == 'train'
    prop_predictor = model
    prop_predictor.train() if training else prop_predictor.eval()
    
    #change_by_Afia
    #for g in optimizer.param_groups:
    #    g['lr'] = g['lr']/sqrt(prop_predictor.current_epoch)
    
    #print(g['lr'])
    if write_path is not None:
        write_file = open(write_path, 'w+')
    stats_tracker = data_utils.stats_tracker(stat_names)

    batch_split_idx = 0
    all_pred_logits, all_labels = [], []  # Used to compute Acc, AUC
    for batch_idx, batch_data in enumerate(tqdm.tqdm(data_loader, dynamic_ncols=True)):
        if training and batch_split_idx % args.batch_splits == 0:
            optimizer.zero_grad()
        batch_split_idx += 1

        #Afia: changed labels_list to MW and MS

        smiles_list, MW_labels_list, MS_labels_list, path_tuple = batch_data

        path_input, path_mask = path_tuple
        
        if args.use_paths:
            path_input = path_input.to(args.device)
            path_mask = path_mask.to(args.device)

        n_data = len(smiles_list)
        mol_graph = MolGraph(smiles_list, args, path_input, path_mask)

        pred_logits = prop_predictor(mol_graph, stats_tracker).squeeze(1)
               
        #print("Printing the type of pred_logits:")
        #print(type(pred_logits))
        #print(list(pred_logits.size())[0])
        #print(pred_logits)
        #print(pred_logits.size())
        
        
        #Afia: adding it  # for backward:
        
        #g = nn.Sigmoid()
        #gate=g(pred_logits)

        #data=reverse_prediction(n_data,pred_logits,smiles_list)

        #Afia: adding it  # for forward
        pred_logits,pred_logits_relued=mask_prediction_by_mass(n_data,pred_logits,smiles_list)
        
        #raw_prediction = (gate * pred_logits + (1. - gate) * data)
        #gate*pred_logits 
        #raw_prediction2= 1. - gate
        #raw_prediction3=raw_prediction2*data
        #(gate * pred_logits + (1. - gate) * data)
        
        #print("Printing raw prediction concatenation:")
        #print(gate)
        #print(raw_prediction)
        #print(raw_prediction3)
        #print(pred_logits)
        
        #pred_logits=raw_prediction
        
        #print(pred_logits)
        #sys.exit()

        #print(pred_logits)
        #print("Printing size of pred logits:")
        #print(pred_logits.size())
        #print(pred_logits_relued)
        #print(pred_logits_relued.size())
        
        #print(pred_logits_relued)
        #predicted=pred_logits.tolist()
        
        #print(MS_labels_list[0])
        #true_logits=make_mass_intensity_pairs(pred_logits,MS_labels_list,n_data)
        #print(true_logits[0][12])
        #print(true_logits[0][13])
        #print(true_logits[0][14])
        #print(true_logits[0][15])
        #print(true_logits.size())
                
        #Afia: checked upto here
        
        true_logits=torch.zeros((n_data,1000), dtype=torch.float64, device = args.device)
        true_logits=make_mass_intensity_pairs(true_logits,MS_labels_list,n_data)
        #print("Printing true_logits")
        #print(type(true_logits))
        #print(type(pred_logits))
        #print(true_logits.size())

        #MS_labels = torch.tensor(MS_labels_list, device=args.device) 
        #print("Size of MS_labels:")
        #print(MS_labels_list[0]) 
        #New thing to implement:
        #declare a tensor of the desired size(same as of pred_logits, supply the argument device=args.device during initialization)
        #replace the value from MS_labels_list to there. #check whether or not it solves your problem.if necessary make a new function.

        if args.loss_type == 'ce':  # memory issues
            all_pred_logits.append(pred_logits)
            all_labels.append(labels)

        if args.loss_type == 'mse':
            loss = nn.MSELoss()(input=pred_logits.float(), target=true_logits.float())
        elif args.loss_type=="generalized_mse":
            loss=make_loss(pred_logits_relued,true_logits)
        elif args.loss_type == 'ce':
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels)
        else:
            assert(False)
        
        
        #print("Printing loss before:")
        #print(loss)
        stats_tracker.add_stat('loss', loss.item() * n_data, n_data)
        #stats_tracker.add_stat('loss', loss * n_data, n_data)
        loss = loss / args.batch_splits
        #print(args.batch_splits)

        if args.loss_type == 'mae':
            mae = torch.mean(torch.abs(pred_logits - labels))
            stats_tracker.add_stat('mae', mae.item() * n_data, n_data)

        if args.loss_type == 'generalized_mse':
            generalized_mse = loss
            stats_tracker.add_stat('generalized_mse', generalized_mse.item() * n_data, n_data)



        #loss.requires_grad = True
        #print("Printing loss before:")
        #print(loss)
        if training:
            #loss.retain_grad() #not sure if I should use this?
            loss.backward()
            #print(model.W_p_o.weight.grad)  #Afia: here the output should not be None. here mine is showing None, something is broken, changed it, now it's not None
            #print(model.embs[0].weight.grad)
            #print("Inside if")
            #print(loss)
            #sys.exit()
            if batch_split_idx % args.batch_splits == 0:
                train_utils.backprop_grads(
                    prop_predictor, optimizer, stats_tracker, args)
                batch_split_idx = 0
        #print(write_path)
        if write_path is not None:
            #Afia: maybe need to change here:
            write_utils.write_props(write_file, smiles_list, MS_labels_list,
                                    pred_logits.cpu().numpy())

    if training and batch_split_idx != 0:
        train_utils.backprop_grads(model, optimizer, stats_tracker, args)  # Any remaining

    if args.loss_type == 'ce':
        all_pred_logits = torch.cat(all_pred_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        pred_probs = nn.Sigmoid()(all_pred_logits).detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy()
        acc = train_utils.compute_acc(pred_probs, all_labels)
        auc = train_utils.compute_auc(pred_probs, all_labels)
        stats_tracker.add_stat('acc', acc, 1)
        stats_tracker.add_stat('auc', auc, 1)

    if write_path is not None:
        write_file.close()
    return stats_tracker.get_stats()


if __name__ == '__main__':
    main()
