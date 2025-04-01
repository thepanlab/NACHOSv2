#!/bin/bash
source /home/pcallec/anaconda3/etc/profile.d/conda.sh
conda activate nachosv2
NACHOSv2_train --loop "cross-validation" --device "cuda:0" --file /home/pcallec/NACHOS/nachosv2/training/training_sequential/config_files/OCT_small/oct_small_1.yml --parallel