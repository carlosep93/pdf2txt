#!/bin/bash
#SBATCH --job-name=translate_xml
#SBATCH --output=slurm_logs/translate_%j.out
#SBATCH --error=slurm_logs/translate_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --qos acc_bscls
#SBATCH -N1
#SBATCH --account bsc88
##SBATCH --qos=acc_bscls
##SBATCH --dependency=afterok:4588757


source /gpfs/scratch/bsc88/bsc088662/envs/use_env_nextpro.sh

umask 007

model_dir='/gpfs/scratch/bsc88/bsc088662/nextpro/model/nllb-200-distilled-600M/'
data_dir='/gpfs/projects/bsc88/data/03-text-repository/03.2-processing/clean/gl'
output_dir='output/gl-es'
src_lang='glg_Latn'
tgt_lang='spa_Latn'
tok_lang='Galician'

tsv_file='nextprocurement_mx_20240626_16.clean'

python translate_xml_nllb.py \
    --model_dir $model_dir \
    --data_dir  $data_dir \
    --output_dir $output_dir \
    --tsv_file $tsv_file \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --tok_lang $tok_lang 


