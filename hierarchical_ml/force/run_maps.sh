#!/bin/sh
#SBATCH --job-name=binary_maps
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siddanib@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=01:00:00
#SBATCH --output=binary_mdl_maps.txt

module load pytorch/1.10
i=0

   for j in 40 100
   do
       for k in $(seq 0.1 0.1 0.4)
       do
           let "i+=1"
           python3 map_generation.py --fl_tag "$i" --re_no $j --phi $k
       done       
   done
