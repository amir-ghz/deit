#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1 
#SBATCH --gres=gpu:2 
#SBATCH --ntasks-per-node=1 
#SBATCH --ntasks=1 
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

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
echo
source activate deit

# Here we are going to run the PyTorch super_resolution example from the PyTorch examples
# GitHub Repository: https://github.com/pytorch/examples/tree/master/super_resolution

# Run PyTorch Training
echo "Training DeiT Start:"
time srun --nodes=1 --gres=gpu:2 --ntasks=per-node=1 --output=deit.out python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /datasets/ImageNet2012nonpub --output_dir checkpoint/deit_tiny
echo

# You're done!
echo "Ending script..."
date

