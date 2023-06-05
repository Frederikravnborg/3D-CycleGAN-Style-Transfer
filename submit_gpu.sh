#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o torch_%J.out
#BSUB -e torch_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1,16
#BSUB -u s214617@dtu.dk
#BSUB -B
#BSUB -R "rusage[mem=2G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1440
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.10.1-python-3.9.16

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source 02466/bin/activate

python train.py
