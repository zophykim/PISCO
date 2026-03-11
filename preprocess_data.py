#preprocess and generate jsonl file
import pandas as pd
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import torch, os, argparse, sys, json, shutil
from tqdm import tqdm
from src.utils import *

esmfold_ckpt_path = 'facebook/esmfold_v1'
tokenizer = AutoTokenizer.from_pretrained(esmfold_ckpt_path)
esmfold_model = EsmForProteinFolding.from_pretrained(esmfold_ckpt_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esmfold_model = esmfold_model.to(device)
esmfold_model.eval()

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Provide the path of the table.csv (including all protein sequences, species, etc.), the working directory and the path to store the jsonl file'
    )
    parser.add_argument('--input_csv', 
                        required = True, 
                        help = 'the path to the table.csv')
    parser.add_argument('--working_dir', 
                        required = False,
                        default = 'tmp/', 
                        help = 'the path to store the intermediate files, e.g. predicted pdb structures. All intermediate files will be removed after the jsonl file is generated.')
    parser.add_argument('--jsonl_path', 
                        required = True, 
                        help = 'the path to store the generated jsonl file, with all features to be used by the model')
    return parser.parse_args()

class dna_protein_sample():
    def __init__(self, protein_seq = None, pdb_path = None):
        self.protein_seq = protein_seq
        self.pdb_path = pdb_path

def main():
    args = parse_args()
    if(not os.path.isfile(args.input_csv)):
        print(f"[ERROR] file not exist: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.working_dir, exist_ok = True)
    df_seq_table = pd.read_csv(args.input_csv, keep_default_na = False)
    with open(args.jsonl_path, 'w', encoding = 'utf-8') as f:
        for idx, row in tqdm(df_seq_table.itertuples(), total = len(df_seq_table)):
        # for idx, row in tqdm(df_seq_table.iterrows(), total = len(df_seq_table)):
            sample_i = dna_protein_sample(protein_seq = row['protein_seq'][:-1], pdb_path = os.path.join(args.working_dir, str(idx) + '.pdb'))
            get_ESMFold_predicted_pdbs(tokenizer, esmfold_model, sample_i.protein_seq, sample_i.pdb_path, device)
            backbone_coords = extract_backbone_coordinates(sample_i.pdb_path)
            secondary_structure_code = extract_secondary_structure_code(sample_i.pdb_path)
            os.remove(sample_i.pdb_path)
            record = {'name': row['ID'], 
                      'organism': row['organism'], 
                      'protein_seq': row['protein_seq'], 
                      'dna_seq': row['dna_seq'], 
                      'protein_coords': backbone_coords, 
                      'protein_secstru': secondary_structure_code, }
            f.write(json.dumps(record, ensure_ascii = False) + '\n')
    #shutil.rmtree(args.working_dir)

if __name__ == '__main__':
    main()

