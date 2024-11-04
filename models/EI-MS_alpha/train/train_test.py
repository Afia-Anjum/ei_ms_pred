import torch
import torch.nn as nn
import numpy as np
import tqdm
from math import sqrt


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from arguments import get_args, create_dirs
from models.prop_predictor import PropPredictor
from datasets.mol_dataset import get_loader
from graph.mol_graph import MolGraph
from utils import data_utils, train_utils, write_utils
import pdb

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt



def init_model(args, n_classes):
    prop_predictor = PropPredictor(args, n_classes=n_classes)
    prop_predictor.to(args.device)

    optimizer = torch.optim.Adam(prop_predictor.parameters(), lr=args.lr)
    return prop_predictor, optimizer

def mask_prediction_by_mass(n_data,pred_logits,smiles_list):
    #print(list(pred_logits.size())[0])
    #print(ExactMolWt(Chem.MolFromSmiles('CCOC(=O)COC(=O)C(F)(F)Cl')))
    MW_list=[]
    for i in range(len(smiles_list)):
        p=smiles_list[i]
        MW_list.append(int(ExactMolWt(Chem.MolFromSmiles(p))))
    for i in range(n_data):
        for j in range(1000):
            if j > (MW_list[i]-1):
                pred_logits[i][j]=0.0
    
    m = nn.ReLU()
    pred_logits_relued=m(pred_logits)
    return pred_logits_relued, pred_logits_relued

def make_mass_intensity_pairs(true_logits,MS_labels_list,n_data):
    for i in range(n_data):
        peak=MS_labels_list[i]
        p=peak.split(";")
        peak_locs=[]
        peak_intensities=[]
        for l in range(len(p)-1):
            loc=int(p[l].split()[0])
            intensity=float(p[l].split()[1])
            true_logits[i][loc]=intensity
    return true_logits

def make_weights(true_logits):
    num_bins=true_logits.shape[1]
    weights=np.power(np.arange(1, num_bins + 1),0.5)[np.newaxis, :]
    return weights/np.sum(weights)

def make_loss(pred_logits_relued,true_logits):
    weights=make_weights(true_logits)
    a=torch.abs(pred_logits_relued - true_logits)
    a=a*a
    weights=weights.argmax()
    weighted_square_error=a*weights
    t=torch.mean(weighted_square_error)
    return t


def load_datasets(raw_data, split_idx, args):
    data_splits = data_utils.read_splits('%s/split_%d.txt' % (args.data, split_idx))
    dataset_loaders = {}
    if not args.test_mode:
        dataset_loaders['train'] = get_loader(
            raw_data, data_splits['train'], args, shuffle=False)
        dataset_loaders['valid'] = get_loader(
            raw_data, data_splits['valid'], args, shuffle=False)
    dataset_loaders['test'] = get_loader(
        raw_data, data_splits['test'], args, shuffle=False)
    return dataset_loaders
    
def run_epoch(data_loader, model, optimizer, stat_names, args, mode,
              write_path=None):
    training = mode == 'train'
    prop_predictor = model
    prop_predictor.train() if training else prop_predictor.eval()

    if write_path is not None:
        write_file = open(write_path, 'w+')
    stats_tracker = data_utils.stats_tracker(stat_names)

    batch_split_idx = 0
    all_pred_logits, all_labels = [], []  # Used to compute Acc, AUC
    for batch_idx, batch_data in enumerate(tqdm.tqdm(data_loader, dynamic_ncols=True)):
        if training and batch_split_idx % args.batch_splits == 0:
            optimizer.zero_grad()
        batch_split_idx += 1
        
        smiles_list, MW_labels_list, MS_labels_list, path_tuple = batch_data
        path_input, path_mask = path_tuple
        if args.use_paths:
            path_input = path_input.to(args.device)
            path_mask = path_mask.to(args.device)

        n_data = len(smiles_list)
        mol_graph = MolGraph(smiles_list, args, path_input, path_mask)

        pred_logits = prop_predictor(mol_graph, stats_tracker).squeeze(1)

        pred_logits,pred_logits_relued=mask_prediction_by_mass(n_data,pred_logits,smiles_list)
        true_logits=torch.zeros((n_data,1000), dtype=torch.float64, device = args.device)
        true_logits=make_mass_intensity_pairs(true_logits,MS_labels_list,n_data)


        if args.loss_type == 'ce':  # memory issues
            all_pred_logits.append(pred_logits)
            all_labels.append(labels)

        if args.loss_type == 'mse' or args.loss_type == 'mae':
            loss = nn.MSELoss()(input=pred_logits, target=labels)
        elif args.loss_type=="generalized_mse":
            loss=make_loss(pred_logits_relued,true_logits)
        elif args.loss_type == 'ce':
            pred_probs = nn.Sigmoid()(pred_logits)
            loss = nn.BCELoss()(pred_probs, labels)
        else:
            assert(False)

        stats_tracker.add_stat('loss', loss.item() * n_data, n_data)
        loss = loss / args.batch_splits

        if args.loss_type == 'mae':
            mae = torch.mean(torch.abs(pred_logits - labels))
            stats_tracker.add_stat('mae', mae.item() * n_data, n_data)

        if args.loss_type == 'generalized_mse':
            generalized_mse = loss
            stats_tracker.add_stat('generalized_mse', generalized_mse.item() * n_data, n_data)



        if training:
            loss.backward()
            if batch_split_idx % args.batch_splits == 0:
                train_utils.backprop_grads(
                    prop_predictor, optimizer, stats_tracker, args)
                batch_split_idx = 0
        if write_path is not None:
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
    
def train_model(dataset_loaders, model, optimizer, stat_names, selection_stat,
                train_func, args):
    best_model_path = '%s/model_best' % (args.model_dir)

    train_pred_output = open('%s/train_pred_stats.csv' % args.output_dir, 'w+', buffering=1)

    for file in [train_pred_output]:
        file.write(','.join(stat_names) + '\n')

    if best_model_path != '':
        model.load_state_dict(torch.load(best_model_path))
        print('Loading model from %s' % best_model_path)
    
    with torch.no_grad():
        train_pred_stats = train_func(data_loader=dataset_loaders['train'], model=model, optimizer=None,stat_names=stat_names,mode='test',args=args, write_path='%s/train_results' % args.result_dir)
    print(data_utils.dict_to_pstr(train_pred_stats, header_str='Test:'))
    train_pred_output.write(data_utils.dict_to_dstr(train_pred_stats, stat_names) + '\n')
    train_pred_output.close()

    return train_pred_output, best_model_path


def main():
    args = get_args()
    model_types = ['conv_net', 'conv_net_attn', 'transformer']
    assert args.model_type in model_types
    
    
    if args.multi:
        raw_data = data_utils.read_smiles_multiclass('%s/raw.csv' % args.data)
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
            args=args)

    
if __name__ == '__main__':
    main()
