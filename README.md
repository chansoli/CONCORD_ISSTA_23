# CONCORD_ISSTA23
Code for ISSTA'23 paper "CONCORD: Clone-aware Contrastive Learning for Source Code"

## Environment Setup
```bash
conda create -n concord Python=3.8.12;
conda activate concord;

# install torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge;

# install apex
git clone https://github.com/NVIDIA/apex.git;
cd apex/;
git checkout feae3851a5449e092202a1c692d01e0124f977e4;
pip install -v --disable-pip-version-check --no-cache-dir ./;
cd ../

# install pip packages
cd CONCORD_ISSTA23;
conda install -y numpy=1.22.4 scipy scikit-learn
conda install -y scikit-learn
pip install -r requirements.txt;
export PYTHONPATH=$(pwd);
```

## Model Weights and Data

### Link

You could find the pre-trained weights of the main CONCORD model and task-specific data here:

- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8393793.svg)](https://doi.org/10.5281/zenodo.8393793)

### Data Pre-processing 

During CONCORD's pre-training, we need to align the syntax labels with code tokens, which requires the data pre-processing. To avoid the distribution shift, the task-specific fine-tuning data needs to do the same pre-processing with the following two steps:
- Parse the source code with [Tree-sitter](https://github.com/tree-sitter/py-tree-sitter) and tokenize the sequence following the grammar of corresponding programming languages. 
- Sub-tokenize with the pre-trained [BPE model](vocab/multilingual_5k_repo_50k_vocab.model).

You could refer to the following steps and customize your own data processing accordingly, or check out our data samples (finetune_data.zip) for the expected format of the pre-processed code.

__Build Tree-sitter__

```
cd data_processing;
bash build_tree_sitter.sh;
```

__Example-1: Process CodeXGLUE Defects Detection Data__

```
python process_code.py \
    --task_name cxg_vuldetect \
    --input_file <path_to_CodeXGLUE>/Code-Code/Defect-detection/dataset/test.jsonl \
    --output_file cxg_vd_test.csv \
    --spm_model ../vocab/multilingual_5k_repo_50k_vocab.model
```

__Example-2: Process CodeXGLUE Clone Detection Data__

```
python process_code.py \
    --task_name cxg_clone \
    --input_file <path_to_CodeXGLUE>/Code-Code/Clone-detection-POJ-104/dataset/test.jsonl \
    --output_file cxg_clone_test.jsonl \
    --spm_model ../vocab/multilingual_5k_repo_50k_vocab.model
```

Note that `process_code.py` script is just to illustrate the basic data processing steps using CodeXGLUE raw data; you may need to customize the script for your specific goal.

## Full rehersal
### Run setup
It will create conda env, install things, build tree-sitter/apex, download fine_tune data provided by CONCORD work, codebert-base

call `setup.sh`

activate into concord env
```
conda activate concord
```

Then follow one by one for env check.

```bash
python data_processing/convert_to_concord_triplets.py \
  --input data_processing/clone_sample_raw.jsonl \
  --output data_processing/clone_triplets.json \
  --tokenizer vocab/multilingual_5k_repo_50k_vocab.model \
  --max-length 512 \
  --negatives-per-sample 1
```

Reduce num_hidden_layers to make it use less memory for train
```json:downloads/huggingface/codebert-base/config.json
    "num_hidden_layers": 6,
```

```bash
python scripts/run_concord_clone_aware_pretrain.py \
  --model_name_or_path downloads/huggingface/codebert-base \
  --config_name config/concord_pretrain_config.json \
  --train_file data_processing/clone_triplets.json \
  --validation_file data_processing/clone_triplets.json \
  --output_dir outputs/test_clone_pretrain \
  --do_train True \
  --do_eval True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --overwrite_output_dir True \
  --remove_unused_columns False \
  --max_seq_length 16
```

```bash
python data_processing/make_poj104_cc_small.py
```

```bash
python scripts/run_concord_finetune_cc_cxg.py \
  --task poj104 \
  --model_name_or_path outputs/test_clone_pretrain \
  --config_name outputs/test_clone_pretrain/config.json \
  --tokenizer_name downloads/huggingface/codebert-base \
  --train_data_file data_processing/finetune_data/poj104_cc_small/train.jsonl \
  --eval_data_file data_processing/finetune_data/poj104_cc_small/valid.jsonl \
  --test_data_file data_processing/finetune_data/poj104_cc_small/test.jsonl \
  --output_dir outputs/poj104_finetune \
  --block_size 128 \
  --train_batch_size 6 \
  --eval_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --logging_steps 50 \
  --save_steps 5 \
  --overwrite_output_dir \
  --do_train --do_eval --do_test
```

```bash
python data_processing/make_cxg_vd_small.py
```

```bash
python scripts/run_concord_finetune_vd.py \
    --task_name cxg_vd \
    --train_file data_processing/finetune_data/cxg_vd_small/train_func.csv \
    --validation_file data_processing/finetune_data/cxg_vd_small/valid_func.csv \
    --test_file data_processing/finetune_data/cxg_vd_small/test_func.csv \
    --model_name_or_path outputs/test_clone_pretrain \
    --config_name outputs/test_clone_pretrain/config.json \
    --tokenizer_name downloads/huggingface/codebert-base \
    --output_dir outputs/cxg_vd_small_finetune \
    --max_seq_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_steps 100 \
    --save_steps 500 \
    --overwrite_output_dir \
    --do_train --do_eval --do_predict
```
