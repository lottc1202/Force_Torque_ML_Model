import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets_global import PointDataset
import math
 
def frc_1b(invar_1b,vectors_1b):
    re_no,phi = torch.split(invar_1b,[1,1],dim=-1)
    re_m = (1-phi)*re_no

    f_1 = (1+(0.15*(re_m**0.687)))/((1-phi)**3)

    f_2 = (5.81*phi)/((1-phi)**3)
    f_2 = f_2 + ((0.48*(phi**0.3333))/((1-phi)**4))

    f_3 = (0.61*(phi**3))/((1-phi)**2)
    f_3 += 0.95
    f_3 = f_3*(phi**3)*re_m

    f = (1-phi)*(f_1+f_2+f_3)
    return f*vectors_1b



if __name__ =="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phi = 0.4
    re = 300/(1-phi)
    invar_1b = torch.as_tensor([[re,phi]],device=device)
    vectors_1b = torch.as_tensor([[1.,0.,0.]],device=device)

    f1 = frc_1b(invar_1b,vectors_1b)
    print(f1)
    print(f1/(1-phi))
    
