#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # RLVR models
    # ["Apertus3-8B-sft-RLVR-105"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_105/hf_actor"
    # ["Apertus3-8B-sft-RLVR-450"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_450/hf_actor"
    # ["Apertus3-8B-sft-RLVR-560"]="/capstor/store/cscs/swissai/infra01/reasoning/models/global_step_560/hf_actor"
    # ["Apertus3-8B-sft-RLVR-MR-800"]="/capstor/store/cscs/swissai/infra01/reasoning/models/mr_800/hf_actor"
    ["Apertus3-8B-sft-RLVR-2STG-2000"]="/capstor/store/cscs/swissai/infra01/reasoning/models/2stg_2000/hf_actor"

    # SFT models

    # 8B pre-cooldown
    ["Apertus-8B-7.04T-swissai-tulu-3-adam-deprecate"]="/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft/checkpoint-13446/"
    ["Apertus-8B-7.04T-swissai-tulu-3-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/token-count/Apertus8B-tokens7.04T-it1678000-swissai-tulu-3-sft-0225/checkpoints/6d5f11d2873ecb4d/checkpoint-13446"


    # Chat template ablation
    ["Apertus-8B-7.04T-template-tulu-special-token"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225/checkpoints/9b811fb20bdd09a4/checkpoint-13446"
    ["Apertus-8B-7.04T-temaplate-tulu"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu-swissai-tulu-3-sft-0225/checkpoints/6d5f11d2873ecb4d/checkpoint-13446"

    # 8B-bug baseline
    ["Apertus-8B-7.2T-bug-swissai-tulu-3-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/8b-cooldown-bug-experiments/Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225-max_grad_norm_0.1/checkpoints/206c53edb3c43a3a/checkpoint-13000/"

    # 8B-patched baseline
    ["Apertus-8B-7.2T-patched-swissai-tulu-3-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/baseline-swissai-tulu3-ademamix/Apertus8B-tokens7.2T-it1728000-hotfix-swissai-tulu-3-sft-0225-bs128-lr5e-06-ademamix/checkpoints/06a6eabaa2ec9af4/checkpoint-13446"


    # 8B-bug mixtures
    ["Apertus-8B-7.2T-bug-mixture1-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/dataset-mixtures-fast/Apertus8B-tokens7.2T-it1728000-ademamix-apertus-sft-mixture-1/checkpoints/55212b68b8cb44a9/checkpoint-1622"
    ["Apertus-8B-7.2T-bug-mixture4-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-8b-sweep/dataset-mixtures-fast/Apertus8B-tokens7.2T-it1728000-ademamix-apertus-sft-mixture-4/checkpoints/754c761b6c8b6898/checkpoint-4793"

    # 8B-patched mixture
    ["Apertus-8B-7.2T-patched-mixture1-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/fix-overfit/Apertus8B-tokens7.2T-it1728000-hotfix-apertus-sft-mixture-1-bs512-lr5e-06-epochs1-ademamix/checkpoints/a0db78d43b9ebd41/checkpoint-1622"
    ["Apertus-8B-7.2T-patched-mixture1-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/fix-overfit/Apertus8B-tokens7.2T-it1728000-hotfix-apertus-sft-mixture-1-bs512-lr5e-06-epochs1-ademamix/checkpoints/b0e21de416cd7599/checkpoint-1622"

    ["Apertus-8B-10.2T-newcooldown-mixture1-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-1/checkpoints/7f2faa33edb7f13e/checkpoint-1622"
    ["Apertus-8B-10.2T-newcooldown-mixture2-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-2/checkpoints/44532e711f8d5bee/checkpoint-3914"
    ["Apertus-8B-10.2T-newcooldown-mixture3-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-3/checkpoints/7a61ceff935a6765/checkpoint-4577"
    ["Apertus-8B-10.2T-newcooldown-mixture4-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-4/checkpoints/8b8a3a8c41a2697e/checkpoint-4792"
    ["Apertus-8B-10.2T-newcooldown-mixture1-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-1-ademamix/checkpoints/ee969b526b1995f7/checkpoint-1622"
    ["Apertus-8B-10.2T-newcooldown-mixture2-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-2-ademamix/checkpoints/53e2ead7db07314d/checkpoint-3914"
    ["Apertus-8B-10.2T-newcooldown-mixture3-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-3-ademamix/checkpoints/8df0b5c9349b8015/checkpoint-4577"
    ["Apertus-8B-10.2T-newcooldown-mixture4-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-4-ademamix/checkpoints/1ce7109f7dde7ed2/checkpoint-4792"

    # 70B baseline
    ["Apertus-70B-15T-swissai-tulu-3-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/token-count/Apertus70B-tokens15T-it1155828-ademamix-swissai-tulu-3-sft-0225/checkpoints/c9b2910640c220b1/checkpoint-13446"
#    ["Apertus-70B-15T-swissai-tulu-3-ademamix-new"]=Coming soon... @Skander

    # 70B mixtures
    ["Apertus-70B-15T-mixture1-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-1/checkpoints/53d4338a0a4a4834/checkpoint-12976"
    ["Apertus-70B-15T-mixture2-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-2/checkpoints/03e6f5ffc256bd37/checkpoint-31312"
    ["Apertus-70B-15T-mixture3-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-3/checkpoints/e5c6472115344007/checkpoint-36620"

    # 70B fast mixtures
    ["Apertus-70B-15T-mixture1-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-1/checkpoints/30f93081a9efa4fa/checkpoint-1622"
    ["Apertus-70B-15T-mixture2-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-2/checkpoints/d1efecec81835dd2/checkpoint-3914"
    ["Apertus-70B-15T-mixture3-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-3/checkpoints/416753d72bb96b59/checkpoint-4577"
    ["Apertus-70B-15T-mixture4-fast-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-70b-sweep/dataset-mixtures-fast/Apertus70B-tokens15T-it1155828-ademamix-apertus-sft-mixture-4/checkpoints/7f9593ea352b3b54/checkpoint-4792"

    # 70 fix fast mixtures
    ["Apertus-70B-15T-mixture1-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/fix-overfit/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adam/checkpoints/1a5e69e74fb30840/checkpoint-1622"
    ["Apertus-70B-15T-mixture1-fast-ademamix-fixed"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/apertus-sft-mixture-1-fast-ademamix/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-maxgnorm1-epochs1-ademamix/checkpoints/4fc579918b550aac/checkpoint-1622"
    ["Apertus-70B-15T-mixture4-fast-adam"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/mixture-fast/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-4-bs512-lr2e-06-maxgnorm1-epochs1-adamw_torch/checkpoints/da28d82fb9346ca9/checkpoint-4792"
    ["Apertus-70B-15T-mixture4-fast-ademamix-fixed"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/fix-ademamix-fast-noquote/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-4-bs512-lr2e-06-maxgnorm1-epochs1-ademamix/checkpoints/0bc386f8352f3273/checkpoint-4792"


)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}

# SFT model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "Apertus SFT models"
