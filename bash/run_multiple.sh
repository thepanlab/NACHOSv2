#!/bin/bash
source /home/pcallec/anaconda3/etc/profile.d/conda.sh
conda activate nachosv2
NACHOSv2_train --loop "cross-validation" --device "cuda:0" "cuda:1" --file /home/pcallec/NACHOS/nachosv2/training/training_sequential/training_sequential.py --parallel