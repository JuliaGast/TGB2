#!/bin/bash

#SBATCH --account=def-rrabba
##SBATCH --account=def-bengioy
#SBATCH --time=2-00:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=4           # CPU cores/threads
#SBATCH --gres=gpu:1                # number of GPU(s) per node
#SBATCH --mem=32G                   # memory (per node)
#SBATCH --job-name=TGB2_THG_software_1
#SBATCH --output=outlog/%x-%j.log


SEED=1
mem_dim=16
time_dim=16
emb_dim=16
e_emb_dim=32



python -u examples/linkproppred/thgl-software/tgn.py --seed "$SEED" --e_emb_dim "$e_emb_dim"