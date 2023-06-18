#!/bin/bash

#SBATCH -J E_cat_avg
#SBATCH --partition batch
#SBATCH --nodelist=vll3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH -o slurm/logs/%A-%x.out
#SBATCH -e slurm/logs/%A-%x.err
#SBATCH --time=4-0

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=$SLURM_NNODES

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

seeds=(1 21 42 3473 10741 32450 93462 85015 64648 71950 87557 99668 55552 4811 10741)

export JAVA_HOME=dirname $(readlink -f $(which java))
export PATH=$PATH:$JAVA_HOME

for i in seeds
do
    python main.py \
    --seed ${seeds[$i]} \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 60 \
    --language koBert \
    --audio HuBert \
    --lr 0.00005 \
    --dropout 0.5 \
    --world_size $WORLD_SIZE \
    --rank $SLURM_PROCID
done


