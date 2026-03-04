from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import torch
from Bio.PDB import PDBParser
from MDAnalysis.analysis.dssp import DSSP
import MDAnalysis as mda
import numpy as np

def convert_structure_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def get_ESMFold_predicted_pdbs(tokenizer, esmfold, protein_seq, protein_pdb_path, device):
    tokenized_protein_seq = tokenizer([protein_seq], return_tensors = 'pt', add_special_tokens = False)['input_ids']
    tokenized_protein_seq = tokenized_protein_seq.to(device)
    with torch.no_grad():
        predicted_structure = esmfold(tokenized_protein_seq)
    predicted_pdb = convert_structure_to_pdb(predicted_structure)
    with open(protein_pdb_path, 'w+') as f:
        f.write(''.join(predicted_pdb))

def round_2decimal(item):
	return round(float(item), 3)

def extract_backbone_coordinates(pdb_file_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_file_path)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return {}
    backbone_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C') and residue.has_id('O'):
                    try:
                        n_atom = residue['N']
                        ca_atom = residue['CA']
                        c_atom = residue['C']
                        o_atom = residue['O']
                        n_coords = list(np.round(n_atom.get_coord(), 3))
                        ca_coords = list(np.round(ca_atom.get_coord(), 3))
                        c_coords = list(np.round(c_atom.get_coord(), 3))
                        o_coords = list(np.round(o_atom.get_coord(), 3))
                        n_coords = list(map(round_2decimal, n_coords))
                        ca_coords = list(map(round_2decimal, ca_coords))
                        c_coords = list(map(round_2decimal, c_coords))
                        o_coords = list(map(round_2decimal, o_coords))
                        residue_id = f"{chain.id}{residue.id[1]}{residue.id[2].strip()}"
                        backbone_coords[residue_id] = [n_coords, ca_coords, c_coords, o_coords]
                    except KeyError as e:
                        print(f"Missing atom in residue {residue.id}: {e}")
                        continue  # Skip to the next residue if an atom is missing
    return list(backbone_coords.values())

def extract_secondary_structure_code(pdb_file_path):
    u = mda.Universe(pdb_file_path)
    return ''.join(DSSP(u).run().results.dssp[0])

