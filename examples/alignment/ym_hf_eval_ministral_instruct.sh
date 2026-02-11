MODEL=Ministral-3-OB
CKPT_PATH=mistralai/Ministral-3-8B-Instruct-2512
sbatch --job-name eval-$MODEL scripts/ym_evaluate_hf.sbatch $CKPT_PATH $MODEL
