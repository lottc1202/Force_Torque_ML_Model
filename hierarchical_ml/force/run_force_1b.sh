#!/bin/sh


   for i in {1..17}
   do
      python3 testing_force_1b.py --fl_no $i --req_type 'test'
   done

