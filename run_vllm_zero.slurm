#!/bin/bash

##SBATCH --partition=hitz-exclusive
##SBATCH --account=hitz-exclusive
##SBATCH --qos=regular
#SBATCH --qos=xlong
##SBATCH --qos=test
#SBATCH --time=4-00:00:00
##SBATCH --time=0-00:10:00
#SBATCH --job-name=vllm_qwq
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100-sxm4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512GB
#SBATCH --gres=gpu:4        # num of gpus: 2 gpu 
#SBATCH --output=.slurm/output_%x.log
#SBATCH --error=.slurm/error_%x.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=joseba.fernandezdelanda@ehu.eus 


# "all2_dev", "all2_train", "all2_test"
#DS_IZENA=se_all_dev


#source /scratch/jfernandezde/envs/stance/bin/activate
source /scratch/jfernandezde/envs/vllm/bin/activate

# google/gemma-2-9b-it 
# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen2.5-7B-Instruct

# google/gemma-2-27b-it
# meta-llama/Llama-3.3-70B-Instruct
# Qwen/Qwen2.5-72B-Instruct
# Qwen/QwQ-32B

export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# /scratch/jfernandezde/stance/data/all2_dev.jsonl \
# /scratch/jfernandezde/stance/data/all2_test.jsonl \

python zero-shot-vllm.py \
	--model Qwen/QwQ-32B \
	--in_paths \
		/scratch/jfernandezde/stance/data/all2_dev.jsonl \
		/scratch/jfernandezde/stance/data/all2_test.jsonl \
		/scratch/jfernandezde/stance/data/all2_train.jsonl \
	--num_gpus 4

