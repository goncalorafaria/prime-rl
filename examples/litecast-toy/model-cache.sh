#!/usr/bin/env bash
# Source inside the container, before importing Hugging Face libraries.
export HF_HOME=${LITECAST_MODEL_CACHE:-/tmp/litecast-hf}
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_XET_CACHE=$HF_HOME/xet
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE"
echo "MODEL_CACHE path=$HF_HUB_CACHE datasets=$HF_DATASETS_CACHE"
