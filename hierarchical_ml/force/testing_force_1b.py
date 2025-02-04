import argparse
import os
import numpy as np
import math
import sys
import torch
from torch.utils.data import DataLoader
import scipy.io as sio

from force_1b import *
from datasets_global import PointDataset

# Change the default values in the parser according to requirements 
parser = argparse.ArgumentParser()
parser.add_argument("--fl_no",type=int,default=1,help="file number")
parser.add_argument("--req_type",type=str,default='test',help='Type of data considered')
parser.add_argument("--k_fold",type=int,default=5,help="K-fold cross-validation")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

# Considered datasets
fl_lst =[opt.fl_no]

# Deciding to use GPUs or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

# To store predictions
inv_np = []
vel_np = []
fk_np = []
rl_np = []

# Cross-validation loop
 
for fld_i in range(opt.k_fold):

    global_train_lst = []
    global_test_lst = []

    for fl_no in fl_lst:
        # Total number of samples in each dataset
        fl_nm = "data/"+str(fl_no)+"_bx_sz_LN_26"
        data_matrix = np.loadtxt(fl_nm)
        dt_list = range(np.size(data_matrix,0))
        # Figuring out test samples
        test_sz = int(len(dt_list)/opt.k_fold)
        test_lst = range(fld_i*test_sz,(fld_i+1)*test_sz)
    
        train_val_lst = [x for x in dt_list if x not in test_lst]
    
        # Divide this into 80-20
        train_lst = train_val_lst[:int(0.8*len(train_val_lst))]
        val_lst = train_val_lst[int(0.8*len(train_val_lst)):]

        global_train_lst.append(train_lst)
        global_test_lst.append(test_lst)

    # Creating DataLoader for testing
    test_dataloader = DataLoader(
      PointDataset(global_test_lst,fl_lst=fl_lst,
      num_neig_2b=2,
      num_neig_3b=2),
      batch_size=opt.batch_size,
      shuffle=False)

    # Creating DataLoader for training
    train_dataloader = DataLoader(
      PointDataset(global_train_lst,fl_lst=fl_lst,
      num_neig_2b=2,
      num_neig_3b=2),
      batch_size=opt.batch_size,
      shuffle=False)

    if opt.req_type == 'test':
        req_dataloader=test_dataloader
    else:
        req_dataloader=train_dataloader

    with torch.no_grad():
        for i,datas in enumerate(req_dataloader):
            invar_1b = datas['invar_1b'].to(device)
            vectors_1b = datas['vectors_1b'].to(device)
 
            fake = (frc_1b(invar_1b,vectors_1b))
            real = datas['frc'].to(device)
            
            # Model inputs
            inv_np.append(invar_1b)
            vel_np.append(vectors_1b)
            fk_np.append(fake.detach())
            rl_np.append(real.detach())

# To evaluate average test performance
inv_np = torch.cat(inv_np,dim=0)
vel_np = torch.cat(vel_np,dim=0)
fk_np = torch.cat(fk_np,dim=0)
rl_np = torch.cat(rl_np,dim=0)

inv_np = np.float32(inv_np.cpu())
vel_np = np.float32(vel_np.cpu())
fk_np = np.float32(fk_np.cpu())
rl_np = np.float32(rl_np.cpu())
thisdict = {'fake':fk_np,'real':rl_np,
            'invar_1b':inv_np,'vectors_1b':vel_np}
sio.savemat(str(opt.fl_no)+"_"+opt.req_type+
            "_results_1b.mat",thisdict)
