#!/bin/sh
#SBATCH --job-name=two_body
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siddanib@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=02-00:00:00
#SBATCH --output=binary_mdl_10.txt

module load pytorch/1.10

python3 training_force_2b.py --n_epochs 1000 --num_lyrs 6 --lyr_wdt 50 --num_neig 10

