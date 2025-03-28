#!/bin/bash
source /home/pcallec/anaconda3/etc/profile.d/conda.sh
conda activate nachosv2
python /home/pcallec/NACHOS/nachosv2/training/training_sequential/training_sequential.py --loop "cross-validation" --device "cuda:0" --file /home/pcallec/NACHOS/nachosv2/training/training_sequential/config_files/OCT_small/oct_small_1.yml --parallel