#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2  
#SBATCH --gres=gpu:2 
#SBATCH --ntasks=2 
#SBATCH --output=logs.out 
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=2
#SBATCH --output=deit-%J.out
#SBATCH --job-name=deit



# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
module load anaconda/anaconda3
module load cuda/cuda-11.6

# List the modules that are loaded
module list


nvidia-smi
echo
source activate /home/aghazizadeh/my-envs/torch

time srun 

# Run PyTorch Training
echo "Training DeiT Start:"
time srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 512 --data-path /datasets/ImageNet2012nonpub --output_dir checkpoint/deit_tiny
echo

# You're done!
echo "Ending script..."
date

