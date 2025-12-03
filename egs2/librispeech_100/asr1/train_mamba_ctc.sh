#!/bin/bash
#SBATCH --job-name=proj_train
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/projtrain_%j.out
#SBATCH --error=logs/projtrain_%j.err

source /ocean/projects/cis250187p/twu13/final-project/espnet/tools/activate_python.sh

./run.sh --stage 11 --stop-stage 11