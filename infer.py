import argparse
from email import parser
import torch
import pisco.models
import torch_geometric
import pisco.data
from pisco.data import (
    build_species_codon_probs_with_diagnosis,
    ID2SPECIES
)
from pisco.models import PISCO_Model, PISCO_Config, PISCO_AR_Model
import numpy as np
import csv
import os
import tqdm
import src.evaluation as ev
import warnings
warnings.filterwarnings("ignore")

CODON_USAGE_PATH = "./codon_frequencies_kazusa.jsonl"

def load_model(checkpoint_path, device):
    is_AR = False
    # try loading training state if present
    training_state = os.path.join(checkpoint_path, 'training_state.pt')
    if os.path.exists(training_state):
        state = torch.load(training_state, map_location=device, weights_only=False)
        best_val = state.get('best_val', float('inf'))
        print(f"[INFO] infer from {training_state}, best_val={best_val}")
    # load config if present
    config_path = os.path.join(checkpoint_path, 'config.json')
    config = PISCO_Config.from_json_file(config_path)
    if config.architectures[0] == 'PISCO_AR_Model':
        model = PISCO_AR_Model.from_pretrained(checkpoint_path)
        is_AR = True
    else:
        model = PISCO_Model.from_pretrained(checkpoint_path)
    model.to(device)
    if config.use_species_distribution:
        codon_usage_path = CODON_USAGE_PATH
        probs, diagnostics = build_species_codon_probs_with_diagnosis(jsonl_path=codon_usage_path)
        model.set_species_codon_probs(probs)
        print(f"[INFO] set species codon probs from {codon_usage_path}")

    model.eval()
    return model,is_AR

def infer(model, dataset, device, label_mode, csv_output, codon_usage_path='',is_AR=False):
    print("Start inference on test set ...")
    data_count = 0
    acc_list = []
    codon_usage_loader = ev.CodonUsageLoader()
    codon_usage_train = codon_usage_loader.load_all_species_codon_frequencies(CODON_USAGE_PATH)


    print(f"\n========== STEP 3: Writing results to {csv_output} ==========")
    with open(csv_output, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "protein", "organism", "len", 
                         "rna", "predicted_rna", 
                         "natural_CSI", "natural_GC%", "natural_CFD", "natural_COUSIN",
                         "predicted_CSI", "predicted_GC%", "predicted_CFD", "predicted_COUSIN", 
                         "DTW_distance", "natural_score", "predicted_score"])
        with torch.inference_mode():
            natural_CSI_list = []
            natural_CFD_list = []
            pred_CSI_list = []
            pred_CFD_list = []
            organ_acc = {}
            for i, protein in enumerate(tqdm.tqdm(dataset, ncols=60)):
                data_count += 1
                try:
                    protein = protein.to(device)
                    h_V = (protein.node_s, protein.node_v)
                    h_E = (protein.edge_s, protein.edge_v)
                    if is_AR:
                        pred_codon,logits = model.infer(
                            h_V, protein.edge_index, h_E, protein.seq, protein.raw_seq,
                            species_id=protein.species_id, secstruct=protein.secstruct,
                            species_name=protein.organ, csv_path=codon_usage_path
                        )
                        pred_codon = pred_codon[0]
                        logits = logits[0]
                    else:
                        logits = model.infer(
                            h_V, protein.edge_index, h_E, protein.seq, protein.raw_seq,
                            species_id=protein.species_id, secstruct=protein.secstruct,
                            species_name=protein.organ, csv_path=codon_usage_path
                        )
                        pred_codon = torch.argmax(logits, dim=-1)
                    pred_rna = dataset.codon_indices_to_rna(protein.raw_seq, pred_codon.cpu().numpy())

                    cu = codon_usage_train.get(protein.organ, None)
                    if not cu:
                        cu = codon_usage_loader.load_codon_usage_from_csv(protein.organ,codon_usage_path)
                    cu = ev.convert_codon_usgage_to_relative_weights(cu,True) if cu else None
                    if label_mode:
                        natural_rna = dataset.codon_indices_to_rna(protein.raw_seq, protein.codon.cpu().numpy())
                        seq_acc = (pred_codon == protein.codon).float().mean().item()
                        acc_list.append(seq_acc)

                        organ = protein.organ
                        if organ not in organ_acc:
                            organ_acc[organ] = []
                        organ_acc[organ].append(seq_acc)

                        natural_score = ev.score_codon_sequence_with_logits(logits, protein.codon.cpu().numpy())

                        natural_CSI = ev.get_calculate_csi(natural_rna, cu) if cu else ""
                        natural_GC = ev.get_gc_percent(natural_rna)
                        natural_CFD = ev.get_cfd(natural_rna, cu, threshold=0.3) if cu else ""
                        natural_COUSIN = ev.get_cousin(natural_rna, cu) if cu else ""
                        natural_MINMAX = ev.get_min_max_percentage(natural_rna, cu) if cu else []

                        natural_CSI_list.append(natural_CSI)
                        natural_CFD_list.append(natural_CFD)
                    else:

                        natural_rna = ""
                        natural_score = ""
                        natural_CSI = ""
                        natural_GC = ""
                        natural_CFD = ""
                        natural_COUSIN = ""
                        natural_MINMAX = []
                        
                    seq_len = len(protein.raw_seq) if protein.raw_seq is not None else 0

                    # organism = ID2SPECIES.get(int(protein.species_id.item()), "Unknown")
                    name = protein.name.item() if hasattr(protein.name, "item") else protein.name

                    pred_score = ev.score_codon_sequence_with_logits(logits, pred_codon.cpu().numpy())

                    pred_GC = ev.get_gc_percent(pred_rna)
                    pred_CFD = ev.get_cfd(pred_rna, cu, threshold=0.3) if cu else ""
                    pred_COUSIN = ev.get_cousin(pred_rna, cu) if cu else ""
                    pred_CSI = ev.get_calculate_csi(pred_rna, cu) if cu else ""
                    pred_MINMAX = ev.get_min_max_percentage(pred_rna, cu) if cu else []
                    natural_pred_dtw = ev.get_dtw(natural_MINMAX, pred_MINMAX) if label_mode else None

                    pred_CSI_list.append(pred_CSI)
                    pred_CFD_list.append(pred_CFD)  
                    writer.writerow([
                        name,
                        protein.raw_seq,
                        protein.organ,
                        seq_len,
                        natural_rna,
                        pred_rna,
                        natural_CSI,
                        natural_GC,
                        natural_CFD,
                        natural_COUSIN,
                        pred_CSI,
                        pred_GC,
                        pred_CFD,
                        pred_COUSIN,
                        natural_pred_dtw,
                        natural_score,
                        pred_score
                    ])
                except Exception as e:
                    print(f"Error processing protein index {i}, name {protein.name}: {e}")
                    continue
    print("========== STEP 3: Inference finished ==========")
    if label_mode and acc_list:
        acc = np.mean(acc_list)

        organ_mean_accs = {org: np.mean(v) for org, v in organ_acc.items()}
        overall_organ_mean = np.mean(list(organ_mean_accs.values()))
        
        csi_kl = ev.calculate_kl_divergence(np.array(natural_CSI_list), np.array(pred_CSI_list))
        cfd_kl = ev.calculate_kl_divergence(np.array(natural_CFD_list), np.array(pred_CFD_list))
        
        print(f"\n========== STEP 4: Metrics calculation ==========")
        print(f"Total count: {data_count}, Mean sequence accuracy: {acc:.4f}")
        print("\nPer-organ accuracy:")
        for org, m in organ_mean_accs.items():
            print(f"  {org}: {m:.4f}")
        print(f"\nOrgan-level averaged accuracy: {overall_organ_mean:.4f}")
        print(f"CSI KL Divergence: {csi_kl:.4f}")
        print(f"CFD KL Divergence: {cfd_kl:.4f}")

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return acc
    return None

def save_results(results, output_path):
    np.save(output_path, results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./checkpoint/finetune_seed0', help='Path to the model checkpoint')
    parser.add_argument('--test_input', default='./data/CodonTransformer_test_filtered.jsonl', help='Path to the test input dataset.if pdb_mode is set, should be a CSV file')
    parser.add_argument('--test_output', default='result/temp.csv', help='Path to save the inference results as CSV')
    parser.add_argument('--codon_usage_path', default='./Codon_Usage_kazusa.csv', help='Path to the codon usage CSV file')
    parser.add_argument('--label_mode', action='store_true', help='If set, compute accuracy and output true rna')
    parser.add_argument('--pdb_mode',   action='store_true', help='If set, use PDB data for inference, for example, CSV input with PDB paths')

    args = parser.parse_args()

    print("\n========== STEP 1: Loading model ==========")
    print(f"Loading model from {args.checkpoint} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model,is_AR = load_model(args.checkpoint, device)
    print("Model loaded.")
    print("\n========== STEP 2: Loading test data ==========")
    print(f"Loading test data from {args.test_input} ...")
    if args.pdb_mode:
        test_data = pisco.data.ProteinCsvLazyDataset(args.test_input)
    else:
        test_data = pisco.data.LazyProteinCodonGraphDataset(args.test_input)
    print(len(test_data))
    infer(model, test_data, device, args.label_mode, args.test_output, args.codon_usage_path,is_AR)
    

if __name__ == "__main__":
    main()
