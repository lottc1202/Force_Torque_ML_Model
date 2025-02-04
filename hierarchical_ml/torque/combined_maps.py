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
parser.add_argument("--fl_tag",type=str)
parser.add_argument("--re_no",type=float,default=50.0,help="Reynolds number of the flow")
parser.add_argument("--phi",type=float,default=0.2,help="Reynolds number of the flow")
parser.add_argument("--k_fold",type=int,default=5,help="K-fold cross-validation")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

# Deciding to use GPUs or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

# Prefixed neighbor location
n_xyz = torch.as_tensor([[[1.1,0.0,0.0]]],device=device)

# Create a uniform grid along x and y
grd = torch.linspace(-3,3,151,device=device)

x,y,z = torch.meshgrid(grd,grd,grd,indexing="ij")

# Reshape x,y,z
x = x.reshape(-1,1,1)
y = y.reshape(-1,1,1)
z = z.reshape(-1,1,1)

xyz = torch.cat([x,y,z],dim=-1)

unit_vec = torch.as_tensor([[[1.0,0.0,0.0]]],device=device)

invar_3b = torch.as_tensor([[[opt.re_no,opt.phi]]],device=device)

# To store predictions
fk_np = []
xyz_np = []
invar_np = []
n_xyz_np = []

# Cross-validation loop
 
for fld_i in range(opt.k_fold):
    mdl_2b = trq_2b(lyr_wdt=100,
                    num_lyrs=5,
                    act1=torch.nn.ReLU(),
                    act2=torch.tanh)

    load_path2b = "saved_models/"+"bx_sz_LN_26"
    load_path2b = load_path2b + "_neig_"+str(26)
    load_path2b = load_path2b + "_kfold_"+str(opt.k_fold)+"_"+str(fld_i)
    load_path2b = load_path2b + "_l_"+str(5)+"_w_"+str(100)
    load_path2b = load_path2b+"_torque_2b.tar"

    st_dct2b = torch.load(load_path2b,map_location=torch.device('cpu'))
    mdl_2b.load_state_dict(st_dct2b['mdl'])
    mdl_2b.train(mode=False)
    mdl_2b.to(device)

    mdl_3b = trq_3b(lyr_wdt=150,
                    num_lyrs=5,
                    act1=torch.nn.ReLU(),
                    act2=torch.tanh)

    load_path3b = "saved_models/"+"bx_sz_LN_26"
    load_path3b = load_path3b + "_neig_"+str(10)
    load_path3b = load_path3b + "_kfold_"+str(opt.k_fold)+"_"+str(fld_i)
    load_path3b = load_path3b + "_l_"+str(5)+"_w_"+str(150)
    load_path3b = load_path3b+"_torque_3b.tar"

    st_dct3b = torch.load(load_path3b,map_location=torch.device('cpu'))
    mdl_3b.load_state_dict(st_dct3b['mdl'])
    mdl_3b.train(mode=False)
    mdl_3b.to(device)

    with torch.no_grad():
        for i in range(0,xyz.size(dim=0),100):
            k = min(xyz.size(dim=0)-i,100)

            xyz_1 = torch.narrow(xyz,0,i,k)
            n_xyz_1 = n_xyz.repeat(k,1,1)
            invar_3b_1 = invar_3b.repeat(k,1,1)
            unit_vec_1 = unit_vec.repeat(k,1,1)

            c1 = torch.cat([n_xyz_1,xyz_1],dim=-1)
            c2 = torch.cat([xyz_1,n_xyz_1],dim=-1)

            r_n_xyz = torch.norm(n_xyz_1,p=2,dim=-1,keepdim=True)
            r_xyz_1 = torch.norm(xyz_1,p=2,dim=-1,keepdim=True)

            c3 = torch.where(r_n_xyz<r_xyz_1,c1,c2)

            vectors_3b = torch.cat([unit_vec_1,c3],dim=-1)

            # Also including the influence of each 
            # neighbors binary influence
            vectors_1b = unit_vec_1.squeeze(dim=1)
            vectors_2b_1 = torch.cat([unit_vec_1,n_xyz_1],dim=-1)
            vectors_2b_2 = torch.cat([unit_vec_1,xyz_1],dim=-1)
            invar_2b_1 = invar_3b_1

            fake = (mdl_3b(invar_3b_1,vectors_3b)+
                    mdl_2b(vectors_1b,invar_2b_1,vectors_2b_1)+
                    mdl_2b(vectors_1b,invar_2b_1,vectors_2b_2))
            
            # Model inputs
            fk_np.append(fake.detach())
            xyz_np.append(xyz_1.squeeze(1))
            n_xyz_np.append(n_xyz_1.squeeze(1))
            invar_np.append(invar_3b_1.squeeze(1))

# To save data
fk_np = torch.cat(fk_np,dim=0)
xyz_np = torch.cat(xyz_np,dim=0)
n_xyz_np = torch.cat(n_xyz_np,dim=0)
invar_np = torch.cat(invar_np,dim=0)


fk_np = np.float32(fk_np.cpu())
xyz_np = np.float32(xyz_np.cpu())
n_xyz_np = np.float32(n_xyz_np.cpu())
invar_np = np.float32(invar_np.cpu())


thisdict = {'fake':fk_np,'xyz':xyz_np,
            'n_xyz':n_xyz_np,
            'inpts':invar_np}
sio.savemat("combined_trq_inflc_map_"+opt.fl_tag
             +".mat",thisdict)
