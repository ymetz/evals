MODEL=Ministral-8B-Instruct-2410
CKPT_PATH=mistralai/Ministral-8B-Instruct-2410
sbatch --job-name eval-$MODEL scripts/ym_evaluate_hf.sbatch $CKPT_PATH $MODEL
