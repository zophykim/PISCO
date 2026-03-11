# PISCO

**P**rotein **S**tructure **I**nformed **S**pecies-specific **C**odon **O**ptimization

PISCO is a deep learning model for **species-aware codon optimization** that integrates:

- protein sequence
- protein 3D structure
- organism-specific codon usage

to generate optimized synonymous codon sequences.

## Model Architecture

![PISCO Architecture](assets/model.png)
---

# Installation

## 1. Internal Users (Sensecore Server)

```bash
conda activate /ai/share/workspace/wwtan/my_conda_env/PISCO
```

---

## 2. Standard Installation

### Create environment

```bash
conda create -n pisco python=3.10 -y
conda activate pisco
```

### Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
--index-url https://download.pytorch.org/whl/cu124
```

### Install PyTorch Geometric

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
-f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

# Model Checkpoints

Pretrained models are hosted on Hugging Face.

### Finetuned Model

https://huggingface.co/zero9998/PISCO-finetune

### Pretrained Model

https://huggingface.co/zero9998/PISCO-pretrain

Example usage:

```bash
--checkpoint zero9998/PISCO-finetune
```

The model will automatically download from Hugging Face.

---

# Inference

The goal of inference is to generate an **optimized RNA codon sequence** for a given protein.

Two modes are supported depending on whether reliable structures are available.

---

# Case 1: No Reliable Structure

### Step 1: Preprocess protein sequences

```bash
python preprocess_data.py \
--input_csv data/Rubisco_AlphaFold_database.csv \
--jsonl_path data/Rubisco_AlphaFold_database.jsonl
```

Notes:

- `pdb_path` column is optional
- preprocessing takes about **10 seconds per protein**

---

### Step 2: Run inference

```bash
python infer.py \
--checkpoint zero9998/PISCO-finetune \
--test_input data/Rubisco_AlphaFold_database.jsonl \
--test_output result/Rubisco_result.csv
```

The predicted RNA sequences will appear in:

```
predicted_rna
```

column of the output CSV.

---

# Case 2: Reliable Structures Available

If high-quality structures exist, inference can be performed directly.

Input CSV must contain:

```
pdb_path
```

column pointing to `.pdb` files.

Run:

```bash
python infer.py \
--checkpoint zero9998/PISCO-finetune \
--test_input data/Rubisco_AlphaFold_database.csv \
--test_output result/Rubisco_result.csv \
--pdb_mode
```

---

# Output Metrics

The output CSV contains:

| Column | Description |
|------|------|
| predicted_rna | predicted optimized RNA |
| predicted_score | model preference score |
| predicted_CSI | codon similarity index |
| predicted_GC% | GC content |
| predicted_CFD | codon frequency distribution |
| predicted_COUSIN | codon usage similarity |

When comparing different protein sequences:

**Higher `predicted_score` indicates a better codon sequence according to the model.**

---

# Training

## Pretraining

```bash
python run_hf.py --train
```

With species distribution:

```bash
python run_hf.py --train --use-sd
```

---

## Finetuning

```bash
python finetune.py \
--pretrained ./models_hf/pretrain_xxx
```

Species distribution version:

```bash
python finetune.py \
--pretrained ./models_hf/pretrain_xxx \
--use-sd
```

---

# Project Structure

```
PISCO
│
├─ data/
│  ├─ pdb/
│  ├─ dataset_test.jsonl
│  ├─ Rubisco_AlphaFold_database.csv
├─ pisco/
│  ├─ models
│  ├─ data
├─ result/
├─ src/
├─ codon_frequencies_kazusa.jsonl
├─ Codon_Usage_kazusa.csv
├─ infer.py
├─ preprocess_data.py
├─ run_hf.py(TODO)
├─ finetune.py(TODO)
└─ requirements.txt
```

---

# Citation

If you use PISCO in your research, please cite the corresponding paper.