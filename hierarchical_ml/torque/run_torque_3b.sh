#!/bin/sh
#SBATCH --job-name=three_bd_t
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siddanib@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=06-00:00:00
#SBATCH --output=trinary_mdl_5.txt

module load pytorch/1.10

      python3 training_torque_3b.py --n_epochs 1000 --num_lyrs 5 --lyr_wdt 150 --num_neig 5

      for i in {1..8}
      do
         python3 testing_torque_3b.py --fl_no $i --num_lyrs 5 --lyr_wdt 150 --num_neig 5 --req_type 'train'
         python3 testing_torque_3b.py --fl_no $i --num_lyrs 5 --lyr_wdt 150 --num_neig 5 --req_type 'test'
      done

       for k in {10..17}
       do
          python3 testing_torque_3b.py --fl_no $k --num_lyrs 5 --lyr_wdt 150 --num_neig 5 --req_type 'train'
          python3 testing_torque_3b.py --fl_no $k --num_lyrs 5 --lyr_wdt 150 --num_neig 5 --req_type 'test'
       done
