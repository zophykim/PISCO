# PISCO
Protein structure Informed Species-specific Codon Optimization

## Get environment ready
1. For internal users (sensecore server)
   ```sh
    conda activate /ai/share/workspace/wwtan/my_conda_env/PISCO
   ```

2. For all users
   ```sh
    # Create Conda Environment
    conda create -n project_env python=3.10 -y
    conda activate project_env

    # Install PyTorch (CUDA 12.4)
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

    # Install PyG dependencies
    pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

    # Install remaining packages
    pip install -r requirements.txt
   ```

## Steps to inference the optimized codon sequence
### Case1: No reliable structure
1. Preprocess the protein sequence data
   ```sh
    python preprocess_data.py --input_csv data_demo/Rubisco_AlphaFold_database.csv --jsonl_path data_demo/Rubisco_AlphaFold_database.jsonl
   ```
The pdb_path column can be omitted in this case.
Each sample (protein sequence) requires approximately 10 seconds to process.

2. Inference the optimized codon sequence
   ```sh
    python infer.py --checkpoint 'chekpoint/pretrain_2025-11-26 06:10:40_seed0_subepoch153_sd' --test_input data_demo/Rubisco_AlphaFold_database.jsonl --test_output data_demo/Rubisco_AlphaFold_database_243_label.csv
   ```
The predicted RNA sequences can be found int he output table.

### Case2: Reliable structures are available
1. Inference the optimized codon sequence directly
   ```sh
    python infer.py --checkpoint 'chekpoint/pretrain_2025-11-26 06:10:40_seed0_subepoch153_sd' --test_input data_demo/Rubisco_AlphaFold_database.csv --test_output data_demo/Rubisco_AlphaFold_database_232_label.csv --pdb_mode
   ```
The pdb_path column is required in this case, containing the path to your pdb file.
The predicted RNA sequences can be found int he output table.

Note: When comparing protein sequences, refer to the 'predicted_score' column. The model prefers sequences with higher scores.

## Steps to training model

### pretrain

   ```sh
    python run_hf.py --train 

    # species distribution version:
    python run_hf.py --train --use-sd
    
   ```

### finetune

   ```sh
    python finetune.py --pretrained './models_hf/pretrain_2025-11-26 06:10:40_seed0_subepoch2_sd'
    # species distribution version:  
    python finetune.py --pretrained './models_hf/pretrain_2025-11-26 06:28:37_seed42_subepoch3'
   ```


