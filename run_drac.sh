#!/bin/bash

#SBATCH --account=def-rrabba
##SBATCH --account=def-bengioy
#SBATCH --time=3-00:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=4           # CPU cores/threads
#SBATCH --gres=gpu:1                # number of GPU(s) per node
#SBATCH --mem=32G                   # memory (per node)
#SBATCH --job-name=TGB2_tgn_forum_5
#SBATCH --output=outlog/%x-%j.log


SEED=5

# # EdgeBank
# mem_mode="unlimited"
# python -u examples/linkproppred/thgl-forum/edgebank.py --seed "$SEED" --mem_mode "$mem_mode"

# TGN
mem_dim=16
time_dim=16
emb_dim=16
e_emb_dim=32
python -u examples/linkproppred/thgl-forum/tgn.py --seed "$SEED" --e_emb_dim "$e_emb_dim" \
--mem_dim "$mem_dim" --time_dim "$time_dim" --emb_dim "$emb_dim"
