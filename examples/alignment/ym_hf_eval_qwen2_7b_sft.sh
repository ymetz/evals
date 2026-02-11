MODEL=Olmo-3-7b-instruct-vllm
CKPT_PATH=allenai/Olmo-3-7B-Instruct
sbatch --job-name eval-$MODEL scripts/ym_evaluate_hf.sbatch $CKPT_PATH $MODEL
