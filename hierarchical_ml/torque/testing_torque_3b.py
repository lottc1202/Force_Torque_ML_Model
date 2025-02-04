import argparse
import os
import numpy as np
import math
import sys
import torch
from torch.utils.data import DataLoader
import scipy.io as sio

from torque_2b import *
from torque_3b import *
from datasets_global import PointDataset

# Change the default values in the parser according to requirements 
parser = argparse.ArgumentParser()
parser.add_argument("--fl_no",type=int,default=1,help="file number")
parser.add_argument("--num_neig",type=int,default=10,help="Number of sorted neighbors to consider")
parser.add_argument("--num_lyrs",type=int,default=2,help="number of layer in model")
parser.add_argument("--lyr_wdt",type=int,default=100,help="width of each layer")
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
fk_np = []
rl_np = []

# Cross-validation loop
 
for fld_i in range(opt.k_fold):
    mdl_2b = trq_2b(lyr_wdt=100,
                    num_lyrs=5,
                    act1=torch.nn.ReLU(),
                    act2=torch.tanh)

    mdl_3b = trq_3b(lyr_wdt=opt.lyr_wdt,
                    num_lyrs=opt.num_lyrs,
                    act1=torch.nn.ReLU(),
                    act2=torch.tanh)

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

    load_path2b = "saved_models/"+"bx_sz_LN_26"
    load_path2b = load_path2b + "_neig_"+str(26)
    load_path2b = load_path2b + "_kfold_"+str(opt.k_fold)+"_"+str(fld_i)
    load_path2b = load_path2b + "_l_"+str(5)+"_w_"+str(100)
    load_path2b = load_path2b+"_torque_2b.tar"

    st_dct2b = torch.load(load_path2b,map_location=torch.device('cpu'))
    mdl_2b.load_state_dict(st_dct2b['mdl'])
    mdl_2b.train(mode=False)
    mdl_2b.to(device)

    load_path3b = "saved_models/"+"bx_sz_LN_26"
    load_path3b = load_path3b + "_neig_"+str(opt.num_neig)
    load_path3b = load_path3b + "_kfold_"+str(opt.k_fold)+"_"+str(fld_i)
    load_path3b = load_path3b + "_l_"+str(opt.num_lyrs)+"_w_"+str(opt.lyr_wdt)
    load_path3b = load_path3b+"_torque_3b.tar"

    st_dct3b = torch.load(load_path3b,map_location=torch.device('cpu'))
    # print(st_dct3b['epoch'])
    mdl_3b.load_state_dict(st_dct3b['mdl'])
    mdl_3b.train(mode=False)
    mdl_3b.to(device)

    # Creating DataLoader for testing
    test_dataloader = DataLoader(
      PointDataset(global_test_lst,fl_lst=fl_lst,
      num_neig_2b=26,
      num_neig_3b=opt.num_neig),
      batch_size=opt.batch_size,
      shuffle=False)

    # Creating DataLoader for training
    train_dataloader = DataLoader(
      PointDataset(global_train_lst,fl_lst=fl_lst,
      num_neig_2b=26,
      num_neig_3b=opt.num_neig),
      batch_size=opt.batch_size,
      shuffle=False)

    if opt.req_type == 'test':
        req_dataloader=test_dataloader
    else:
        req_dataloader=train_dataloader

    with torch.no_grad():
        for i,datas in enumerate(req_dataloader):
            vectors_1b = datas['vectors_1b'].to(device)                
            invar_2b = datas['invar_2b'].to(device)
            vectors_2b = datas['vectors_2b'].to(device)

            invar_3b = datas['invar_3b'].to(device)
            vectors_3b = datas['vectors_3b'].to(device)
            
            fake = (mdl_2b(vectors_1b,invar_2b,vectors_2b)+
                    mdl_3b(invar_3b,vectors_3b))

            real = datas['trq'].to(device)
            
            # Model inputs
            fk_np.append(fake.detach())
            rl_np.append(real.detach())

# To evaluate average test performance
fk_np = torch.cat(fk_np,dim=0)
rl_np = torch.cat(rl_np,dim=0)


fk_np = np.float32(fk_np.cpu())
rl_np = np.float32(rl_np.cpu())
thisdict = {'fake':fk_np,'real':rl_np}
sio.savemat(str(opt.fl_no)+"_"+opt.req_type+
            "_results_l_"+str(opt.num_lyrs)+
            "_w_"+str(opt.lyr_wdt)+"_3b.mat",thisdict)
