#!/bin/sh
#SBATCH --job-name=two_body_t
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=siddanib@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --mem=100gb
#SBATCH --time=06-00:00:00
#SBATCH --output=binary_mdl_2_6.txt

module load pytorch/1.10

for m in 2 6
do
   for j in {50..100..50}
   do 
      python3 training_torque_2b.py --n_epochs 1000 --num_lyrs $m --lyr_wdt $j --num_neig 26

      for i in {1..8}
      do
         python3 testing_torque_2b.py --fl_no $i --num_lyrs $m --lyr_wdt $j --num_neig 26 --req_type 'train'
         python3 testing_torque_2b.py --fl_no $i --num_lyrs $m --lyr_wdt $j --num_neig 26 --req_type 'test'
      done

       for k in {10..17}
       do
          python3 testing_torque_2b.py --fl_no $k --num_lyrs $m --lyr_wdt $j --num_neig 26 --req_type 'train'
          python3 testing_torque_2b.py --fl_no $k --num_lyrs $m --lyr_wdt $j --num_neig 26 --req_type 'test'
       done
    done
done
