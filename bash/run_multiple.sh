#!/bin/bash
source /home/pcallec/anaconda3/etc/profile.d/conda.sh
conda activate nachosv2
mpirun -n 3 NACHOSv2_train --loop "cross-validation" --device "cuda:0" "cuda:1" --file /home/pcallec/NACHOS/nachosv2/training/training/config_files/OCT_small/oct_small_1.yml