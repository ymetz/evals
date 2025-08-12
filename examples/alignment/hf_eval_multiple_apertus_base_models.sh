#!/bin/bash

export TOKENIZER=alehc/swissai-tokenizer
export BOS=true

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(    
    # Base pretrained models
    ["Apertus8B-tokens7.04T-it1678000"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.04T-it1678000"
    ["Apertus8B-tokens7.2T-it1728000"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.2T-it1728000"
    ["Apertus70B-tokens15T-it1155828"]="/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828"
    ["Apertus70B-tokens15T-long-context-64k"]="/capstor/store/cscs/swissai/infra01/users/ctianche/long-ctx-70B-runs/Meg_Runs/main-long-ctx-runs-70b-v1/apertus3-70b-64k-256nodes-theta12M/checkpoints"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}

# Base model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "Apertus base models"
