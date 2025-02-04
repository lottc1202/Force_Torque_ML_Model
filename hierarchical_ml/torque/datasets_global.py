import glob
import os
import numpy as np
import torch
import math

from torch.utils.data import Dataset, DataLoader

# Dataset class
# method 1: __len__ so that len(dataset) returns the size of the dataset. 
# method 2 __getitem__ to support the indexing such that dataset[i] can be used to get ith sample

class PointDataset(Dataset): # inheritated the class of Pytorch Dataset,
    def __init__(self,ind_lst,fl_lst=[1,2],num_neig_2b=26,
                      num_neig_3b=26):
        super().__init__()
        self.num_neig_2b = num_neig_2b
        self.num_neig_3b = num_neig_3b
        for i,(indices,fl_no) in enumerate(zip(ind_lst,fl_lst)):
            fl_nm = "data/"+str(fl_no)+"_bx_sz_LN_26"
            data_matrix_full = np.loadtxt(fl_nm)
            # Create weight matrix
            wgt_matrix = np.ones((np.size(data_matrix_full,0),1))
            # The number of samples in each dataset is different
            # So including that aspect in wgt_matrix as well
            wgt_matrix = wgt_matrix*math.sqrt(2000.0/np.size(wgt_matrix,0))
            if i == 0:
                self.data_matrix = data_matrix_full[indices]
                self.wgt_matrix = wgt_matrix[indices]
            else:
                self.data_matrix = np.concatenate((self.data_matrix,
                                   data_matrix_full[indices]),axis=0)
                self.wgt_matrix = np.concatenate((self.wgt_matrix,
                                   wgt_matrix[indices]),axis=0)

    def __getitem__(self, index): 
        # numpy arrays (all inputs are non-dimensionalized)
        neig_part = self.data_matrix[index,12:12+7*self.num_neig_2b]
        mn_vel = self.data_matrix[index,0:3]
        mn_phi = self.data_matrix[index,3:4]
        frc = self.data_matrix[index,4:7]
        trq = self.data_matrix[index,7:10]

        # Creating tensors
        neig_part = torch.as_tensor(neig_part,dtype=torch.float)
        mn_vel = torch.as_tensor(mn_vel,dtype=torch.float)
        mn_phi = torch.as_tensor(mn_phi,dtype=torch.float)
        frc = torch.as_tensor(frc,dtype=torch.float)
        trq = torch.as_tensor(trq,dtype=torch.float)

        # Reshape the tensors
        neig_part = torch.reshape(neig_part,(-1,7))
        mn_vel = torch.reshape(mn_vel,(3,))
        mn_phi = torch.reshape(mn_phi,(1,))
        frc = torch.reshape(frc,(3,))
        trq = torch.reshape(trq,(3,))

        # Creating 1-body (mean model) information
        invar_1b = []
        invar_1b.append(torch.norm(mn_vel,p=2,
                       keepdim=True))
        invar_1b.append(mn_phi)
        invar_1b = torch.cat(invar_1b,dim=0)

        vectors_1b = mn_vel/torch.norm(mn_vel,
                             p=2)

        # Creating 2-body (pairwise) information
        invar_2b = []
        vectors_2b = []
        psvectors_2b = []
        for i in range(self.num_neig_2b):
            r_i = neig_part[i,0:3]
            # Creating invariants of each neighbor
            inpt_1 = torch.norm(mn_vel[0:],p=2)
            inpt = torch.stack([inpt_1,mn_phi[0]],dim=0)
            invar_2b.append(inpt)
            # Creating vectors of each neighbhor
            vec_1 = mn_vel/inpt_1
            vec_2 = r_i
            vecs = torch.cat([vec_1,vec_2],
                               dim=0)
            vectors_2b.append(vecs)

            # Creating pseudo-vectors
            psvec = torch.cross(vec_1,r_i,dim=0)
            psvectors_2b.append(psvec)

        invar_2b = torch.stack(invar_2b,dim=0)
        vectors_2b = torch.stack(vectors_2b,dim=0)
        psvectors_2b = torch.stack(psvectors_2b,dim=0)

        # Creating 3-body (trinary) information
        invar_3b = []
        vectors_3b = []
        psvectors_3b = []
        for i in range(self.num_neig_3b-1):
            r_i = neig_part[i,0:3]
            for j in range(i+1,self.num_neig_3b): 
                r_j = neig_part[j,0:3]
                # Creating invariants for each combination
                inpt_1 = torch.norm(mn_vel[0:],p=2)
                inpt = torch.stack([inpt_1,mn_phi[0]],dim=0)
                invar_3b.append(inpt)

                # Creating vectors for each combination
                vec_1 = mn_vel/inpt_1
                vec_2 = r_i
                vec_3 = r_j
                vecs = torch.cat([vec_1,vec_2,
                                  vec_3],
                                   dim=0)
                vectors_3b.append(vecs)

                # Creating pseudo-vectors
                psvec_1 = torch.cross(vec_1,r_i,dim=0)
                psvec_2 = torch.cross(vec_1,r_j,dim=0)
                psvec_3 = torch.cross(r_i,r_j,dim=0)

                psvecs = torch.cat([psvec_1,psvec_2,
                                    psvec_3],dim=0)

                psvectors_3b.append(psvecs)

        invar_3b = torch.stack(invar_3b,dim=0)
        vectors_3b = torch.stack(vectors_3b,dim=0)
        psvectors_3b = torch.stack(psvectors_3b,dim=0)

        # Create a dictionary to return
        data_dct = {}
        data_dct['invar_1b'] = invar_1b
        data_dct['vectors_1b'] = vectors_1b
        data_dct['invar_2b'] = invar_2b
        data_dct['vectors_2b'] = vectors_2b
        data_dct['psvectors_2b'] = psvectors_2b
        data_dct['invar_3b'] = invar_3b
        data_dct['vectors_3b'] = vectors_3b
        data_dct['psvectors_3b'] = psvectors_3b 
        data_dct['frc'] = frc
        data_dct['trq'] = trq
        data_dct['wgt_f'] = (torch.ones(1)*
                            self.wgt_matrix[index,0]*
                            self.data_matrix[index,10])
        data_dct['wgt_t'] = (torch.ones(1)*
                             self.wgt_matrix[index,0]*
                             self.data_matrix[index,11])
        return data_dct

    def __len__(self):
        return np.size(self.data_matrix,0)


if __name__ =="__main__":
    dataloader = DataLoader(
      PointDataset([[1,2,],],fl_lst=[7,],num_neig_2b=4,num_neig_3b=3),
      batch_size=2,
      shuffle=True)
    for i,datas in enumerate(dataloader):
        print(datas['trq'].shape)
        print(datas['psvectors_2b'])
        print(datas['invar_3b'])
        print(datas['psvectors_3b'])
        print(datas['wgt_t'])
        break
