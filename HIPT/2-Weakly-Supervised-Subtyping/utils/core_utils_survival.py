import numpy as np
import torch
import torch.nn.functional as F
from utils.utils_survival import *
import os
import torch.nn.functional as F
from datasets_CLAM.dataset_generic import save_splits
from models.model_dsmil import *
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_dgcn import DeepGraphConv
from models.model_clam_survival import CLAM_MB, CLAM_SB
from models.model_cluster import MIL_Cluster_FC
from models.model_hierarchical_mil import HIPT_None_FC, HIPT_LGP_FC, HIPT_GP_FC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index 
import pandas as pd
import sys
#from utils.gpu_utils import gpu_profile, print_gpu_mem
#os.environ['GPU_DEBUG']='0'

def coxph_loss(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    sort_time = torch.argsort(time, 0, descending=True)
    event = torch.gather(event, 0, sort_time)
    
    risk = torch.gather(y_pred, 0, sort_time)
    exp_risk = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(exp_risk, 0))
    censored_likelihood = (risk - log_risk) * event
    censored_likelihood = torch.sum(censored_likelihood)
    censored_likelihood = censored_likelihood / y_time.shape[0]
    return -censored_likelihood
def nll_loss(hazards, S, Y, c, alpha=0.15, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss
class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss
def c_index(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    return concordance_index(time, -np.exp(y_pred), event)
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif  self.best_score - score>0.0005:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience or epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))

    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0,10.0]).to('cuda'))
    print('Done!')
    print('\nSetup EarlyStopping...', end=' ')

    print('\nInit Model...', end=' ')
    model_dict = {'path_input_dim': args.path_input_dim, "dropout": args.drop_out, 'n_classes': args.n_classes,'feature_type': args.features}
    # model_dict = { "dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0,10.0]).to('cuda'))

        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    elif 'hipt' in args.model_type:
        if args.model_type == 'hipt_n':
            model = HIPT_None_FC(**model_dict)
        elif args.model_type == 'hipt_lgp':
            model = HIPT_LGP_FC(**model_dict, freeze_4k=args.freeze_4k, pretrain_4k=args.pretrain_4k, freeze_WSI=args.freeze_WSI, pretrain_WSI=args.pretrain_WSI)
        elif args.model_type == 'hipt_gp':
            model = HIPT_GP_FC(**model_dict, freeze_WSI=args.freeze_WSI, pretrain_WSI=args.pretrain_WSI)
    elif args.model_type == 'dsmil':
        i_classifier = FCLayer(in_size=args.path_input_dim, out_size=model_dict['n_classes'])
        b_classifier = BClassifier(input_size=args.path_input_dim, output_class=model_dict['n_classes'], dropout_v=0.0)
        model = MILNet(i_classifier, b_classifier)
    elif args.model_type == 'dgcn':
        model_dict = {'path_input_dim': args.path_input_dim}
        model = DeepGraphConv(num_features=model_dict['path_input_dim'], n_classes=args.n_classes)
    elif args.model_type == 'mi_fcn':
        model = MIL_Cluster_FC(path_input_dim=args.path_input_dim, n_classes=args.n_classes)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 5, stop_epoch=20, verbose = True)

    else:
        early_stopping = None
    print('Done!')
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, dropinput=0.25)
            
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    val_index = summary(model, val_loader, args.n_classes)
    print('Fold:{},Valid:  C-Index: {:.4f}'.format(cur,val_index))

    val_index = summary(model, test_loader, args.n_classes)
    with open(args.results_dir+'/fold_'+str(cur)+'.txt','w') as f:
        f.write('Fold:{},Valid:  C-Index: {:.4f}'.format(cur,val_index)+'\n')
    print('Fold:{},Valid:  C-Index: {:.4f}'.format(cur,val_index))

    if writer:
        writer.add_scalar('val_index', val_index, 0)
    
    writer.close()
    return val_index


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, gc=32):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, batch in enumerate(loader):

        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device, non_blocking=True), cluster_id, label.to(device, non_blocking=True)
        else:
            data, label = batch
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            cluster_id = None

        logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
        #logits, Y_prob, Y_hat, _, _ = model(x_path=data)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss = loss / gc
        loss.backward()

        # step
        
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters"):
                data, cluster_id, label = batch
                data, cluster_id, label = data.to(device, non_blocking=True), cluster_id, label.to(device, non_blocking=True)
            else:
                data, label = batch
                data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
                cluster_id = None
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(x_path=data)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, dropinput=0.0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    
    WSI_logits = []
    slide_time = []
    slide_event = []
    
    Cindex_logits = []
    Cindex_time = []
    Cindex_event = []

    step = 0
    gc = 0
    epochCindex = 0
    epochCoxloss = 0
    for batch_idx, batch in enumerate(loader):
        
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            ###====
            data, label,y_time,y_event = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        if dropinput > 0:
            data = F.dropout(data, p=dropinput)
        # print('================',label)
        logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=cluster_id, label=label, instance_eval=True)
        # logits, Y_prob, Y_hat, _, instance_dict = model(h=data, label=label, instance_eval=True)
###============
        WSI_logits.append(logits[0][0].clone())
        slide_time.append(y_time[0].item())
        slide_event.append(y_event[0].item())
        
        
        Cindex_logits.append(logits[0].item())
        Cindex_time.append(y_time[0].item())
        Cindex_event.append(y_event[0].item())
        step +=1
        if(step!=0 and step%10 ==0):
                
                WSI_logits = torch.stack(WSI_logits)
                # print('=======WSI_logits2222=====',WSI_logits)
                slide_time = torch.tensor(slide_time).to('cuda:0')
                slide_event = torch.tensor(slide_event).to('cuda:0')

                coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
                # print('===coxloss======',coxloss)
                coxloss.backward()
                # step


                epochCoxloss += coxloss
                WSI_logits = []
                slide_time = []
                slide_event = []
                gc +=1
                if(gc!=0 and gc%8 == 0):
                    optimizer.step()
                    optimizer.zero_grad()
    epochCoxloss = epochCoxloss*25/len(loader)
    cindex = c_index(Cindex_logits,Cindex_time,Cindex_event)
    print('Train: Epoch: {}, coxloss: {:.4f} cindex: {:.4f}'.format(epoch,epochCoxloss, cindex))


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        WSI_logits = []
        slide_time = []
        slide_event = []
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters"):
                data, cluster_id, label = batch
                data, cluster_id, label = data.to(device), cluster_id, label.to(device)
            else:
                data, label,y_time,y_event = batch
                data, label = data.to(device), label.to(device)
                cluster_id = None
            logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=cluster_id, label=label, instance_eval=True)
            # logits, Y_prob, Y_hat, _, instance_dict = model(h=data,label=label, instance_eval=True)
            WSI_logits.append(logits[0][0])
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())
            
    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
    cindex = c_index(WSI_logits,slide_time,slide_event)
    print('Valid: Epoch: {}, coxloss: {:.4f}cindex: {:.4f}'.format(epoch,coxloss, cindex))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, coxloss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']

    patient_results = {}
    WSI_logits = []
    slide_time = []
    slide_event = []

    for batch_idx, batch in enumerate(loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            # data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            data, label,y_time,y_event  = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        #data, label = data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(data)
            WSI_logits.append(logits[0][0])
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())
    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
    cindex = c_index(WSI_logits,slide_time,slide_event)
    print('Valid:  coxloss: {:.4f}cindex: {:.4f}'.format(coxloss, cindex))
    return  cindex
def summary_score(fold,model, loader, n_classes):
    results_dir = n_classes.results_dir
    n_classes = n_classes.n_classes 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']

    patient_results = {}
    WSI_logits = []
    slide_time = []
    slide_event = []

    for batch_idx, batch in enumerate(loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            # data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            data, label,y_time,y_event  = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        #data, label = data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(data)
            WSI_logits.append(logits[0][0].cpu())
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())
    # print('==============',logits[0][0])
    final_score = pd.DataFrame({'slide_ids': slide_ids,  'val_score' : np.array(WSI_logits)})
    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
    cindex = c_index(WSI_logits,slide_time,slide_event)
    print('Valid:  coxloss: {:.4f}cindex: {:.4f}'.format(coxloss, cindex))
    final_score.to_csv(os.path.join(results_dir,'fold_'+str(fold)+'.csv' ))
    return  cindex
