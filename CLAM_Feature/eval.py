### Base Packages
from __future__ import print_function
import argparse
import pdb
import os
import math
import sys

### Numerical Packages
import numpy as np
import pandas as pd

### Internal Imports
from datasets_CLAM.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train,summary
from models.model_clam import CLAM_MB, CLAM_SB
# from utils.eval_utils_survival import *
### PyTorch Imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings
import pdb
warnings.filterwarnings('ignore')
##### Train-Val-Test Loop for 10-Fold CV

def main(args):
    ### Creates Results Directory (if not previously created)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    ### Which folds to evaluates + iterate
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    
    ### 10-Fold CV Loop.
    all_test_auc = []
    all_pred = []
    folds = np.arange(start, end)
    for i in folds:
        print('======Fold '+str(i)+' test begin========')
        seed_torch(args.seed) ### Sets the Torch.Seed
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_0.csv'.format(args.splits))
        datasets = (test_dataset)
        
        ckpt_path = args.weights_path + 's_'+str(i)+'_checkpoint.pt'
        args.path_input_dim = 384

        model = initiate_model(args, ckpt_path)
        loader = get_simple_loader(datasets)
        if(len(loader)>1):
            patient_results, test_error, auc, acc_logger = summary(model, loader, args.n_classes)
            
        if(len(loader)==1):
            patient_results, test_error,  auc, acc_logger,Y_pred = summary(model, loader, args.n_classes)
            all_pred.append(Y_pred)

          
        print('======Fold '+str(i)+' ==========test AUC=========',auc)
        all_test_auc.append(auc)

        ### Writes results to PKL File
        # filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        # save_pkl(filename, results)

    ### Saves results as a CSV file

    final_df = pd.DataFrame({'folds': folds,  'test_auc' : all_test_auc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    if (len(all_pred)>0 and (all_pred.count(0) >= all_pred.count(1))):
        with open(os.path.join(args.results_dir, 'summary.txt'),'w') as f:
            f.write(str(0))
    elif(len(all_pred)>0 and (all_pred.count(0) <= all_pred.count(1))):
        with open(os.path.join(args.results_dir, "summary.txt"),'w') as f:
            f.write(str(1))
##### Argparser
### (Default) Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir',  type=str, default='/media/ssd1/pan-cancer', help='data directory')
parser.add_argument('--max_epochs',     type=int, default=20, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr',             type=float, default=2e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac',     type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg',            type=float, default=1e-5,  help='weight decay (default: 1e-5)')
parser.add_argument('--seed',           type=int, default=123, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k',              type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start',        type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end',          type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir',    type=str, default='./results', help='results directory (default: ./results)')
parser.add_argument('--opt',            type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--bag_loss',       type=str, choices=['svm', 'ce'], default='ce', help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_size',     type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--mode',           type=str, default='path', help='Which features to load')
parser.add_argument('--log_data',       action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing',        action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--drop_out',       action='store_true', default=True, help='enabel dropout (p=0.25)')
parser.add_argument('--weighted_sample',action='store_true', default=False, help='enable weighted sampling')

### CLAM specific options
parser.add_argument('--bag_weight',     type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B',              type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--inst_loss',      type=str, choices=['svm', 'ce', None], default='svm', help='instance-level clustering loss function (default: None)')
parser.add_argument('--no_inst_cluster',action='store_true', default=False, help='disable instance-level clustering')
parser.add_argument('--subtyping',      action='store_true', default=False, help='subtyping problem')

### Options Used
parser.add_argument('--model_type',     type=str, default='clam_sb', help='Type of model to use',
                    choices=['clam_sb', 'clam_mb', 'mil', 'dgcn', 'mi_fcn', 'dsmil', 'hipt_n', 'hipt_lgp'])
parser.add_argument('--features',       type=str, default='vits_tcga_pancancer_dino', help='Which features to use',
                    choices=['resnet50_trunc', 'vits_tcga_pancancer_dino'])
parser.add_argument('--task',           type=str, default='tcga_lung_subtype', help='Which weakly-supervised task to evaluate on.')
parser.add_argument('--path_input_dim', type=int, default=384, help='Size of patch embedding size (384 for DINO)')

parser.add_argument('--prop',           type=float, default=1.0, help='Proportion of training dataset to use')
parser.add_argument('--pretrain_4k',    type=str, default='None', help='Whether to initialize the 4K Transformer in HIPT', choices=['None', 'vit4k_xs_dino'])
parser.add_argument('--splits',           type=str, default='', help='Which features to load')
parser.add_argument('--weights_path',           type=str, default='', help='Which features to load')
parser.add_argument('--feature_path',           type=str, default='', help='Which features to load')
parser.add_argument('--csv_path',           type=str, default='', help='Which features to load')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### Creating Experiment Code
### 1. If HIPT, set the mode to be 'pyramid'

model_code = args.model_type

### 3. Add embedding dimension in the experiment code.
if args.path_input_dim != 384:
    model_code += '_%d' % args.path_input_dim

### 3. Add task information in the experiment code.



print("Setting Splits Directory...", args.splits)

##### Setting the seed + log settings
def initiate_model(args, ckpt_path):
    print('Init Model')

    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
  
    #print_network(model)

    ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

##### Loading the dataset
print('\nLoad Dataset')
print(args.task)
study = "_".join(args.task.split('_')[:2])


if args.task == 'tcga_lung_subtype':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = '/dssg/home/acct-medftn/medftn/BEPT/Model/benchMark/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/TCGA_LUAD.csv',
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            mode=args.mode,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='OSFL',
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            prop=args.prop,
                            ignore=[])
elif args.task == 'tcga_crc_subtype':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path =  '/dssg/home/acct-medftn/medftn/BEPT/Model/benchMark/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/TCGA_CRC.csv',
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            mode=args.mode,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='OSFL',
                            label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
                            patient_strat=False,
                            prop=args.prop,
                            ignore=[])
elif args.task == 'tcga_stad_subtype':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path =  '/dssg/home/acct-medftn/medftn/BEPT/Model/benchMark/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/TCGA_STAD.csv',
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            mode=args.mode,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='OSFL',
                            label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
                            patient_strat=False,
                            prop=args.prop,
                            ignore=[])
elif args.task == 'tcga_ccrcc_subtype':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path =  '/dssg/home/acct-medftn/medftn/BEPT/Model/benchMark/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/TCGA_KIRC.csv',
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            mode=args.mode,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='OSFL',
                            label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
                            patient_strat=False,
                            prop=args.prop,
                            ignore=[])
elif args.task == 'tcga_prcc_subtype':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = '/dssg/home/acct-medftn/medftn/BEPT/Model/benchMark/HIPT/2-Weakly-Supervised-Subtyping/dataset_csv/TCGA_PRCC.csv',
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            mode=args.mode,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='OSFL',
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            prop=args.prop,
                            ignore=[])
elif args.task == 'tcga_brca_subtype':
    args.n_classes = 2
    study_dir = args.feature_path
    
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, study_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_col='oncotree_code',
                            label_dict = {'IDC':0, 'ILC':1},
                            patient_strat=False,
                            ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
    
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if 'subtype' in args.task:
    exp_folder = args.task
args.results_dir = os.path.join(args.results_dir)

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)
else:
    if 'summary.csv' in os.listdir(args.results_dir):
        print("results already exists! Exiting script.")
        import sys
        sys.exit()

print('split_dir: ', args.splits)
assert os.path.isdir(args.splits)

settings.update({'split_dir': args.splits})

with open(args.results_dir + '/experiment.txt', 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    
    results = main(args)
    
    print("finished!")
    print("end script")


