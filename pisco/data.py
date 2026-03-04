import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from collections import defaultdict
import os
from typing import Dict, List, Tuple, Optional
from src.utils import *
import tempfile
import pandas as pd

# ====== Global constants ======
LETTER_TO_NUM = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '_': 20
}
NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}
AA2CODONS = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAU', 'AAC'],
    'D': ['GAU', 'GAC'],
    'C': ['UGU', 'UGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'K': ['AAA', 'AAG'],
    'M': ['AUG'],
    'F': ['UUU', 'UUC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    '_': ['UAA', 'UGA', 'UAG']
}
SPECIES2ID = {
    'Unknown': 0,
    'Nicotiana tabacum': 1,
    'Mus musculus': 2,
    'Danio rerio': 3,
    'Arabidopsis thaliana': 4,
    'Escherichia coli general': 5,
    'Caenorhabditis elegans': 6,
    'Drosophila melanogaster': 7,
    'Chlamydomonas reinhardtii': 8,
    'Saccharomyces cerevisiae': 9,
    'Klebsiella michiganensis': 10,
    'Klebsiella pasteurii': 11,
    'Klebsiella oxytoca': 12,
    'Pseudomonas putida': 13,
    'Raoultella planticola': 14,
    'Klebsiella variicola': 15,
    'Serratia fonticola': 16,
    'Raoultella terrigena': 17,
    'Kosakonia radicincitans': 18,
    'Raoultella ornithinolytica': 19,
    'Klebsiella pneumoniae subsp. pneumoniae HS11286': 20,
    'Klebsiella quasivariicola': 21,
    'Serratia nevei': 22,
    'Klebsiella grimontii': 23,
    'Klebsiella quasipneumoniae': 24,
    'Serratia quinivorans': 25,
    'Serratia plymuthica AS9': 26,
    'Serratia liquefaciens': 27,
    'Citrobacter europaeus': 28,
    'Serratia marcescens': 29,
    'Klebsiella aerogenes': 30,
    'Serratia bockelmannii': 31,
    'Pluralibacter gergoviae': 32,
    'Citrobacter farmeri': 33,
    'Serratia ficaria': 34,
    'Serratia ureilytica': 35,
    'Serratia grimesii': 36,
    'Shigella dysenteriae': 37,
    'Serratia entomophila': 38,
    'Citrobacter freundii': 39,
    'Citrobacter werkmanii': 40,
    'Citrobacter amalonaticus': 41,
    'Enterobacter chengduensis': 42,
    'Citrobacter portucalensis': 43,
    'Citrobacter braakii': 44,
    'Enterobacter ludwigii': 45,
    'Enterobacter quasiroggenkampii': 46,
    'Rouxiella badensis': 47,
    'Enterobacter mori': 48,
    'Serratia rubidaea': 49,
    'Saccharolobus solfataricus': 50,
    'Brenneria goodwinii': 51,
    'Shigella boydii': 52,
    'Ewingella americana': 53,
    'Enterobacter kobei': 54,
    'Shigella sonnei': 55,
    'Bacillus subtilis': 56,
    'Enterobacter cancerogenus': 57,
    'Enterobacter sichuanensis': 58,
    'Yokenella regensburgei': 59,
    'Enterobacter roggenkampii': 60,
    'Leclercia adecarboxylata': 61,
    'Rahnella aquatilis CIP 78.65 = ATCC 33071': 62,
    'Kalamiella piersonii': 63,
    'Enterobacter hormaechei': 64,
    'Citrobacter koseri ATCC BAA-895': 65,
    'Enterobacter asburiae': 66,
    'Pantoea allii': 67,
    'Pectobacterium atrosepticum': 68,
    'Escherichia marmotae': 69,
    'Pectobacterium parmentieri': 70,
    'Escherichia fergusonii': 71,
    'Citrobacter cronae': 72,
    'Pectobacterium versatile': 73,
    'Pectobacterium aroidearum': 74,
    'Pectobacterium polaris': 75,
    'Dickeya fangzhongdai': 76,
    'Yersinia massiliensis CCUG 53443': 77,
    'Yersinia intermedia': 78,
    'Kosakonia cowanii': 79,
    'Pectobacterium carotovorum': 80,
    'Escherichia albertii': 81,
    'Salmonella bongori N268-08': 82,
    'Yersinia alsatica': 83,
    'Dickeya solani': 84,
    'Dickeya dadantii 3937': 85,
    'Erwinia persicina': 86,
    'Pectobacterium brasiliense': 87,
    'Yersinia frederiksenii ATCC 33641': 88,
    'Hafnia alvei': 89,
    'Dickeya dianthicola': 90,
    'Obesumbacterium proteus': 91,
    'Lelliottia amnigena': 92,
    'Hafnia paralvei': 93,
    'Escherichia coli O157-H7 str. Sakai': 94,
    'Pantoea ananatis PA13': 95,
    'Photorhabdus laumondii subsp. laumondii TTO1': 96,
    'Dickeya zeae': 97,
    'Atlantibacter hermannii': 98,
    'Pantoea stewartii': 99,
    'Escherichia coli str. K-12 substr. MG1655': 100,
    'Yersinia kristensenii': 101,
    'Yersinia proxima': 102,
    'Yersinia enterocolitica': 103,
    'Cronobacter dublinensis subsp. dublinensis LMG 23823': 104,
    'Yersinia mollaretii ATCC 43969': 105,
    'Yersinia aleksiciae': 106,
    'Cronobacter sakazakii': 107,
    'Yersinia pseudotuberculosis IP 32953': 108,
    'Cronobacter malonaticus LMG 23826': 109,
    'Yersinia rochesterensis': 110,
    'Providencia rettgeri': 111,
    'Escherichia ruysiae': 112,
    'Providencia stuartii': 113,
    'Yersinia aldovae 670-83': 114,
    'Yersinia rohdei': 115,
    'Yersinia pestis A1122': 116,
    'Pantoea vagans': 117,
    'Providencia heimbachae': 118,
    'Pantoea dispersa': 119,
    'Edwardsiella anguillarum ET080813': 120,
    'Pantoea agglomerans': 121,
    'Proteus mirabilis HI4320': 122,
    'Morganella morganii': 123,
    'Providencia rustigianii': 124,
    'Providencia alcalifaciens': 125,
    'Xenorhabdus bovienii str. feltiae Florida': 126,
    'Proteus terrae subsp. cibarius': 127,
    'Citrobacter youngae': 128,
    'Proteus vulgaris': 129,
    'Proteus penneri': 130,
    'Edwardsiella piscicida': 131,
    'Lonsdalea populi': 132,
    'Edwardsiella tarda': 133,
    'Shigella flexneri 2a str. 301': 134,
    'Erwinia amylovora CFBP1430': 135,
    'Yersinia ruckeri': 136,
    'Plesiomonas shigelloides': 137,
    'Edwardsiella ictaluri': 138,
    'Providencia thailandensis': 139,
    'Moellerella wisconsensis': 140,
    'Enterobacter cloacae': 141,
    'Rosenbergiella epipactidis': 142,
    'Enterobacter bugandensis': 143,
    'Thermococcus litoralis': 144,
    'Thermoccoccus kodakarensis': 145,
    'Serratia proteamaculans': 146,
    'Thermococcus chitonophagus': 147,
    'Pyrococcus furiosus': 148,
    'Thermococcus gammatolerans': 149,
    'Thermococcus barophilus MPT': 150,
    'Thermococcus onnurineus': 151,
    'Thermococcus sibiricus': 152,
    'Pyrococcus horikoshii': 153,
    'Pyrococcus yayanosii': 154,
    'Cronobacter turicensis': 155,
    'Candidatus Hamiltonella defensa 5AT (Acyrthosiphon pisum)': 156,
    'Proteus faecis': 157,
    'Nicotiana tabacum chloroplast': 158,
    'Candidatus Erwinia haradaeae': 159,
    'Buchnera aphidicola (Schizaphis graminum)': 160,
    'Salmonella enterica subsp. enterica serovar Typhimurium str. LT2': 161,
    'Chlamydomonas reinhardtii chloroplast': 162,
    'Pectobacterium odoriferum': 163,
    'Homo sapiens': 164,
    # 'Pichia pastoris': 165,
}
ID2SPECIES = {v: k for k, v in SPECIES2ID.items()}
CODON_TO_INDEX_MAP = {}
INDEX_TO_CODON_MAP = defaultdict(dict)
for aa, codons in AA2CODONS.items():
    for idx, codon in enumerate(codons):
        CODON_TO_INDEX_MAP[codon] = idx
        INDEX_TO_CODON_MAP[aa][idx] = codon

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
                                
class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch



class BatchSampler_iterShuffle(data.Sampler):
    """
    Sampler that batches according to a maximum node count.
    Each __iter__() reshuffles globally and rebuilds batches to ensure strong randomness.

    :param node_counts: list[int], number of nodes for each sample
    :param max_nodes: int, maximum nodes per batch
    :param shuffle: bool, whether to shuffle
    """
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        self.node_counts = node_counts
        # Filter out samples exceeding max_nodes
        self.idx = [i for i in range(len(node_counts))
                    if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes

    def __len__(self):
        # Return an estimate (average batch count)
        est_batch_size = max(1, self.max_nodes // max(self.node_counts))
        return (len(self.idx) + est_batch_size - 1) // est_batch_size

    def __iter__(self):
        # Reshuffle on each iteration
        if self.shuffle:
            random.shuffle(self.idx)

        idx_copy = self.idx.copy()
        while idx_copy:
            batch = []
            n_nodes = 0
            while idx_copy and n_nodes + self.node_counts[idx_copy[0]] <= self.max_nodes:
                next_idx = idx_copy.pop(0)
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            yield batch


class LazyProteinCodonGraphDataset(data.Dataset):
    def __init__(self, jsonl_path, 
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu"
                 ):
        self.path = jsonl_path
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = []
        self.offsets = []
        
        # Generate cache file paths (managed centrally)
        if jsonl_path is not None:
            root = os.path.dirname(jsonl_path)
            file_name = os.path.splitext(os.path.basename(jsonl_path))[0]
            self.node_counts_cache = os.path.join(root, f"{file_name}_node_counts.npy")
            self.offsets_cache = os.path.join(root, f"{file_name}_offsets.npy")

            # Try to load cache
            if not self._try_load_cache(file_name):
               # Recompute if no cache
                self._build_index()
                self._save_cache()
        else:
            self.node_counts_cache = None
            self.offsets_cache = None

    def _try_load_cache(self,file_name):
        """Try to load cache files; return True on success, False on failure"""
        try:
            if os.path.exists(self.node_counts_cache) and os.path.exists(self.offsets_cache):
                self.node_counts = np.load(self.node_counts_cache).tolist()
                self.offsets = np.load(self.offsets_cache).tolist()
                print(f"Loaded index cache: {file_name}")    
                return True
        except Exception as e:
            print(f"Failed to load cache, recomputing: {e}")
            # Clear any partially loaded cache (to avoid dirty data)
            self.node_counts = []
            self.offsets = []
        return False
    
    def _build_index(self):
        """Build file index (node_counts and offsets)"""
        with open(self.path, 'r') as f:
            # First get total number of lines (for tqdm progress bar)
            total_lines = sum(1 for _ in f)
            f.seek(0)
            
            # Initialize index
            self.node_counts = []
            self.offsets = []
            offset = 0

            # Process lines one by one
            for line in tqdm.tqdm(f, total=total_lines, desc="Building index"):
                self.offsets.append(offset)
                offset += len(line.encode('utf-8'))
                
                data = json.loads(line)
                self.node_counts.append(len(data['protein_seq']))

    def _save_cache(self):
        """Save cache to files"""
        np.save(self.node_counts_cache, np.array(self.node_counts))
        np.save(self.offsets_cache, np.array(self.offsets))


    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        with open(self.path, 'r') as f:
            f.seek(self.offsets[i])
            line = f.readline()
            protein = json.loads(line)
        return self._featurize_as_graph(protein)
        
    def _featurize_as_graph(self, protein):
        name = protein['name']
        organ = protein['organism']
        raw_seq = protein['protein_seq']   # This is the original amino acid sequence string
        secstruct = protein['protein_secstru']

        # secstruct encoding: H=0, E=1, -=2
        secstruct_map = {'H': 0, 'E': 1, '-': 2}
        secstruct_tensor = torch.as_tensor([secstruct_map.get(s, 2) for s in secstruct], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if organ not in SPECIES2ID:
                species_id = torch.tensor(SPECIES2ID['Unknown'], dtype=torch.long)
                print(f"Warning: Unknown organism '{organ}' in protein '{name}'. Assigned to 'Unknown' species.")
                # raise ValueError(f"Unknown organism: {organ}")
            else:
                species_id = torch.tensor(SPECIES2ID[organ], dtype=torch.long)
            
            coords = torch.as_tensor(protein['protein_coords'], 
                                     device=self.device, dtype=torch.float32)   
            
            seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['protein_seq']],
                                  device=self.device, dtype=torch.long)
            


            if 'rna_seq' in protein and protein['rna_seq']:
                codon = torch.as_tensor(self.rna_to_codon_indices(protein['rna_seq']),
                                    device=self.device, dtype=torch.long)
            elif 'dna_seq' in protein and protein['dna_seq']:
                codon = torch.as_tensor(self.rna_to_codon_indices(protein['dna_seq'].replace('T', 'U')),
                                    device=self.device, dtype=torch.long)
            else:
                raise ValueError(f"Missing RNA/DNA sequence for protein '{name}'")
            
            #check lengths
            if not (len(seq) == len(codon) == coords.shape[0]+1 == secstruct_tensor.shape[0]+1):
                raise ValueError(f"Length mismatch in protein '{name}': "
                                 f"seq({len(seq)}), codon({len(codon)}), "
                                 f"coords({coords.shape[0]}), secstruct({secstruct_tensor.shape[0]})")


            # === Align lengths ===
            
            last = coords[-1:]
            pad = last.repeat(len(seq) - coords.shape[0], 1, 1)
            coords = torch.cat([coords, pad], dim=0)
            
            last = secstruct_tensor[-1:]
            pad = last.repeat(len(seq) - secstruct_tensor.shape[0])
            secstruct_tensor = torch.cat([secstruct_tensor, pad], dim=0)


            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))

        data = torch_geometric.data.Data(
            x=X_ca, seq=seq, raw_seq=raw_seq, codon=codon, name=name, organ=organ,
            secstruct=secstruct_tensor,
            node_s=node_s, node_v=node_v,
            edge_s=edge_s, edge_v=edge_v,
            edge_index=edge_index, mask=mask, species_id=species_id
        )
        return data

    def rna_to_codon_indices(self, rna_seq):
        codon_indices = []
        for i in range(0, len(rna_seq), 3):
            codon = rna_seq[i:i+3].upper()
            if codon in CODON_TO_INDEX_MAP:
                codon_indices.append(CODON_TO_INDEX_MAP[codon])
            else:
                raise ValueError(f"Invalid or unknown codon: {codon}")
        return codon_indices
    
    def codon_indices_to_rna(self, aa_seq, codon_idx_seq):
        assert len(aa_seq) == len(codon_idx_seq)
        rna = ''
        for aa, idx in zip(aa_seq, codon_idx_seq):
            codon = INDEX_TO_CODON_MAP[aa.upper()].get(idx, None)
            if codon is None:
                raise ValueError(f"No codon index {idx} for amino acid {aa}")
            rna += codon
        return rna
    
    def batch_codon_indices_to_rna(self, aa_seq, codon_indices_batch):
        if isinstance(aa_seq, str):
            aa_seq = [a for a in aa_seq]
        n_samples = len(codon_indices_batch)
        L = len(aa_seq)
        aa_seq = [a.upper() for a in aa_seq]
        rna_list = []
        for codon_indices in codon_indices_batch:
            codons = [INDEX_TO_CODON_MAP[aa_seq[i]][idx] for i, idx in enumerate(codon_indices)]
            rna = ''.join(codons)
            rna_list.append(rna)
        return rna_list

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.linalg.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0, dim=-1), dim=-1)
        # n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        # n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        # perp = _normalize(torch.cross(c, n))
        perp = _normalize(torch.cross(c, n, dim=-1))  # Explicitly specify dim
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

def build_species_codon_probs_with_diagnosis(
    jsonl_path: str,
    species_to_id: Dict[str, int] = SPECIES2ID,
    num_species: int = len(SPECIES2ID),
    codon_to_index_map: Dict[str, int] = CODON_TO_INDEX_MAP,
    index_to_codon_map: Dict[str, Dict[int, str]] = INDEX_TO_CODON_MAP,
    letter_to_num: Dict[str, int] = LETTER_TO_NUM,
    aa2codons: Dict[str, List[str]] = AA2CODONS,
    max_slots: int = 6,
    diagnose_n_species: int = 3,
    diagnose_n_aa_per_species: int = 3,
    specific_checks: Optional[List[Tuple[str, str]]] = [("Thermococcus onnurineus", "M")],  # list of (species_name, aa_letter)
    seed: int = 42,
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict]:
        """
        Read json/jsonl -> build probs tensor, and output diagnostic report.
        Returns: (probs tensor shape (num_species, 21, max_slots), diagnostics dict)

        Parameter explanation (key points):
        - species_to_id: species name -> id (must match your model)
        - codon_to_index_map: e.g. {'GCU':0, 'GCC':1, ...}
        - index_to_codon_map: e.g. {'A': {0:'GCU',1:'GCC',...}, ...} or global mapping by index -> codon
                            (we support both shapes)
        - letter_to_num: AA letter -> aa_id (0..19)
        - aa2codons: aa letter -> list of codons (for fallback)
        - specific_checks: If provided, will prioritize diagnosis for these (species, aa)
                            (usually for cases you want to strongly verify)
        Example diagnostic report is shown in the description below.
        """

        # 1. Load json / jsonl and retain raw counts data structure for diagnostic comparison
        raw_species = {}  # species_name -> {AA_letter: (codons_list, counts_list)}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    # Could be one big dict of many species
                    for sname, sdict in parsed.items():
                        raw_species[sname] = sdict
            except Exception:
                # fallback: linewise jsonl
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # merge keys
                        for sname, sdict in obj.items():
                            raw_species[sname] = sdict

        # 2. Initialize probs tensor
        probs = torch.zeros((num_species, 21, max_slots), dtype=torch.float32)

        # helper: find codon index with U<->T fallback (case-insensitive)
        def find_codon_index(codon: str):
            c = codon.upper()
            if c in codon_to_index_map:
                return codon_to_index_map[c]
            # try swap U/T
            alt = c.replace('U','T') if 'U' in c else c.replace('T','U')
            if alt in codon_to_index_map:
                return codon_to_index_map[alt]
            return None

        # 3. Fill counts into probs (raw counts — we'll normalize later)
        # Also keep a copy of what we parsed for diagnostics
        parsed_counts = {}  # species_name -> aa_letter -> list of (codon, count)
        for sname, sdict in raw_species.items():
            parsed_counts[sname] = {}
            if sname not in species_to_id:
                # keep for diagnostics but skip filling if unmapped
                continue
            sid = int(species_to_id[sname])
            # sdict: e.g. 'A': [["GCU","GCC","GCA","GCG"], [16988,8217,8206,8282]]
            for aa_letter, pair in sdict.items():
                if aa_letter not in letter_to_num:
                    continue
                aa_id = letter_to_num[aa_letter]
                try:
                    codons_list, counts_list = pair[0], pair[1]
                except Exception:
                    # if structure unexpectedly different, try to be robust
                    # e.g., sometimes it might already be dict-like; skip if unknown
                    continue

                # keep a sorted list for diagnostics later
                parsed_counts[sname][aa_letter] = list(zip([c.upper() for c in codons_list], [int(x) for x in counts_list]))

                for codon_str, cnt in zip(codons_list, counts_list):
                    idx = find_codon_index(codon_str)
                    if idx is None:
                        # not mapped -> skip but record in diagnostics
                        continue
                    if idx >= max_slots:
                        continue
                    probs[sid, aa_id, idx] += float(cnt)

        # 4. Normalize / fallback to uniform distribution (if row is all 0)
        diagnostics = {
            "unmapped_species_in_json": [s for s in raw_species.keys() if s not in species_to_id],
            "per_species_aa_total_counts": {},  # optional sums
            "rows_fallback_uniform": [],  # list of (species_id, aa_letter)
            "slot_warnings": []  # list of warnings
        }

        for sid in range(num_species):
            # try find species name(s) for human readable diagnostics
            # reverse lookup optional
            species_names = [k for k,v in species_to_id.items() if int(v) == sid]
            species_name = species_names[0] if species_names else f"<id_{sid}>"
            diagnostics["per_species_aa_total_counts"][species_name] = {}

            for aa_letter, aa_id in letter_to_num.items():
                row = probs[sid, aa_id]
                s = float(row.sum().item())
                # record raw sum
                diagnostics["per_species_aa_total_counts"][species_name][aa_letter] = s
                if s > 0.0:
                    probs[sid, aa_id] = row / s
                else:
                    # fallback: evenly distribute among known codon slots for this aa (via aa2codons & codon_to_index_map)
                    codons = aa2codons.get(aa_letter, [])
                    slots = []
                    for codon in codons:
                        idx = codon_to_index_map.get(codon)
                        if idx is None:
                            alt = codon.replace('U','T') if 'U' in codon else codon.replace('T','U')
                            idx = codon_to_index_map.get(alt)
                        if idx is not None and idx < max_slots:
                            slots.append(idx)
                    slots = sorted(list(set(slots)))
                    if len(slots) == 0:
                        # extreme fallback -> put all mass on slot 0
                        probs[sid, aa_id, 0] = 1.0
                        diagnostics["rows_fallback_uniform"].append((species_name, aa_letter, [0]))
                    else:
                        val = 1.0 / len(slots)
                        for si in slots:
                            probs[sid, aa_id, si] = val
                        diagnostics["rows_fallback_uniform"].append((species_name, aa_letter, slots))

        # 5. Diagnostic print: select several (species, aa) and compare JSON top counts and model slots
        if verbose:
            random.seed(seed)
            species_list_available = [s for s in parsed_counts.keys()]
            # prepare list of checks
            checks = []
            if specific_checks:
                # only include those species that exist in json file
                for sname, aa in specific_checks:
                    if sname not in parsed_counts:
                        continue
                    if aa not in parsed_counts[sname]:
                        # include anyway to show it's missing
                        checks.append((sname, aa))
                    else:
                        checks.append((sname, aa))
            # sample random species/aa if not enough specific
            n_needed = diagnose_n_species
            sampled_species = []
            if len(species_list_available) > 0:
                sampled_species = random.sample(species_list_available, min(diagnose_n_species, len(species_list_available)))
            # build final checks
            for sname in sampled_species:
                # choose AA with largest total counts in that species (if available), else random AA
                aa_counts = parsed_counts.get(sname, {})
                if aa_counts:
                    # sum totals per aa
                    aa_totals = [(aa, sum(cnt for _, cnt in pairs)) for aa, pairs in aa_counts.items()]
                    aa_totals_sorted = sorted(aa_totals, key=lambda x: x[1], reverse=True)
                    top_aas = [aa for aa, _ in aa_totals_sorted[:diagnose_n_aa_per_species]]
                    for aa in top_aas:
                        checks.append((sname, aa))
                else:
                    # fallback: check some common aas
                    some_aas = list(letter_to_num.keys())[:diagnose_n_aa_per_species]
                    for aa in some_aas:
                        checks.append((sname, aa))

            # deduplicate checks while preserving order
            seen = set()
            final_checks = []
            for item in checks:
                if item not in seen:
                    final_checks.append(item)
                    seen.add(item)

            # Print header
            print("=== Species-codon probs loader: DIAGNOSTIC REPORT ===")
            print(f"json file: {jsonl_path}")
            print(f"num species in species_to_id: {num_species}, species mapped in json: {len(parsed_counts)}")
            if diagnostics["unmapped_species_in_json"]:
                print("WARNING: these species present in json but missing in species_to_id (will be ignored):")
                for s in diagnostics["unmapped_species_in_json"]:
                    print("   ", s)
            print()

            # For each check, print comparison
            for sname, aa in final_checks:
                print(f"--- species: {sname} | aa: {aa} ---")
                sid = species_to_id.get(sname, None)
                if sid is None:
                    print("  * species not present in species_to_id -> skipped filling. Still showing json raw counts if present.")
                # JSON raw top counts
                json_pairs = parsed_counts.get(sname, {}).get(aa, [])
                if not json_pairs:
                    print("  JSON: (no counts found for this aa in json)")
                else:
                    # sort by counts desc
                    sorted_json = sorted(json_pairs, key=lambda x: x[1], reverse=True)
                    top3 = sorted_json[:3]
                    print("  JSON top codons (codon, count):")
                    for cod, cnt in top3:
                        print(f"     {cod}  {cnt}")
                # model-probs vector
                if sid is not None:
                    aa_id = letter_to_num[aa]
                    prob_vec = probs[sid, aa_id].tolist()  # length max_slots
                    # show as (slot_idx: prob -> mapped codon if known)
                    print("  Model slots (slot_idx: prob, mapped_codon_if_any):")
                    for slot_idx, p in enumerate(prob_vec):
                        # try find codon label from index_to_codon_map
                        mapped_cod = None
                        # index_to_codon_map might be two shapes: per-aa mapping (aa -> {idx: codon}) or flat idx->codon
                        if isinstance(index_to_codon_map, dict) and aa in index_to_codon_map and slot_idx in index_to_codon_map[aa]:
                            mapped_cod = index_to_codon_map[aa][slot_idx]
                        else:
                            if isinstance(index_to_codon_map, dict):
                                # try both integer and string keys for flat mappings
                                mapped_cod = index_to_codon_map.get(slot_idx)
                        mapped_str = mapped_cod if mapped_cod is not None else "-"
                        print(f"     [{slot_idx}]: {p:.4f}    {mapped_str}")
                    # Now check whether JSON top codons are represented in mapped slots
                    if json_pairs:
                        topcodons = [c for c,_ in sorted(json_pairs, key=lambda x: x[1], reverse=True)][:3]
                        mapped_codons_on_slots = []
                        for slot_idx in range(max_slots):
                            # attempt to find mapping codon label
                            m = None
                            if isinstance(index_to_codon_map, dict) and aa in index_to_codon_map and slot_idx in index_to_codon_map[aa]:
                                m = index_to_codon_map[aa][slot_idx]
                            else:
                                m = index_to_codon_map.get(slot_idx)
                            if m:
                                mapped_codons_on_slots.append(m.upper())
                        # compare sets / order
                        missing = [c for c in topcodons if c.upper() not in mapped_codons_on_slots]
                        if missing:
                            print("  MISMATCH WARNING: JSON top codon(s) not present in model slot mappings:", missing)
                        else:
                            # check order heuristic: whether slot probs reflect JSON order (loose check)
                            print("  JSON top codons are present in model slots.")
                    print()
                else:
                    print()

            # Summary
            if diagnostics["rows_fallback_uniform"]:
                print("Note: fallback uniform rows assigned for these (species,aa) because json had no counts for them:")
                for entry in diagnostics["rows_fallback_uniform"][:10]:
                    print("   ", entry)
            print("=== End of diagnostic ===\n")

        return probs, diagnostics

# class ProteinJsonlGenerator:
#     """
#     Generate JSONL from input CSV (columns: ID, organism, seq, dna, pdb_path).
#     JSONL example:
#     {
#       "name": 13329,
#       "organism": "Arabidopsis thaliana",
#       "seq": "MVKIARTQAKEQCRKRIRNYF_",
#       "dna": "ATGGTAAAGATCGCAAGAACTCAAGCTAAGGAACAATGCAGAAAGAGAATTAGAAACTATTTCTAA",
#       "coords": [[[...]], ...],
#       "secstru": "-HHHHHHHHHHHHHHHHHHH-"
#     }
#     """

class ProteinCsvLazyDataset(LazyProteinCodonGraphDataset):
    """
    Lazy loading Dataset for protein CSV.
    Each __getitem__ extracts PDB features on demand and returns a featurized Data object.
    Input: CSV with columns [ID, organism, protein_seq, dna_seq, pdb_path]
    Output: torch_geometric.data.Data
    """
    def __init__(self, csv_path, device="cpu", **kwargs):
        super().__init__(jsonl_path=None, device=device, **kwargs)
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna('')
        self.length = len(self.df)
        if 'pred' not in self.df.columns:
            self.df['pred'] = ''

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_path = row['pdb_path']
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Missing pdb: {pdb_path}")
        coords = extract_backbone_coordinates(pdb_path)
        secstru = extract_secondary_structure_code(pdb_path)
        protein = {
            "name": row["ID"],
            "organism": row["organism"],
            "protein_seq": row["protein_seq"],
            "dna_seq": row["dna_seq"],
            "protein_coords": coords,
            "protein_secstru": secstru,
        }
        return self._featurize_as_graph(protein)
    
    def _featurize_as_graph(self, protein):
        name = protein['name']
        organ = protein['organism']
        raw_seq = protein['protein_seq']   # This is the original amino acid sequence string
        secstruct = protein['protein_secstru']

        # secstruct encoding: H=0, E=1, -=2
        secstruct_map = {'H': 0, 'E': 1, '-': 2}
        secstruct_tensor = torch.as_tensor([secstruct_map.get(s, 2) for s in secstruct], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if organ not in SPECIES2ID:
                species_id = torch.tensor(SPECIES2ID['Unknown'], dtype=torch.long)
                print(f"Warning: Unknown organism '{organ}' in protein '{name}'. Assigned to 'Unknown' species.")
                # raise ValueError(f"Unknown organism: {organ}")
            else:
                species_id = torch.tensor(SPECIES2ID[organ], dtype=torch.long)
            
            coords = torch.as_tensor(protein['protein_coords'], 
                                     device=self.device, dtype=torch.float32)   
            
            seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['protein_seq']],
                                  device=self.device, dtype=torch.long)
            


            if 'rna_seq' in protein and protein['rna_seq']:
                codon = torch.as_tensor(self.rna_to_codon_indices(protein['rna_seq'].upper()),
                                    device=self.device, dtype=torch.long)
            elif 'dna_seq' in protein and protein['dna_seq']:
                codon = torch.as_tensor(self.rna_to_codon_indices(protein['dna_seq'].upper().replace('T', 'U')),
                                    device=self.device, dtype=torch.long)
            else:
                codon = torch.as_tensor([],
                                    device=self.device, dtype=torch.long)
                
            #check lengths
            if not (len(seq) == coords.shape[0]+1 == secstruct_tensor.shape[0]+1):
                raise ValueError(f"Length mismatch in protein '{name}': "
                                 f"seq({len(seq)}), codon({len(codon)}), "
                                 f"coords({coords.shape[0]}), secstruct({secstruct_tensor.shape[0]})")


            # === Align lengths ===
            
            last = coords[-1:]
            pad = last.repeat(len(seq) - coords.shape[0], 1, 1)
            coords = torch.cat([coords, pad], dim=0)
            
            last = secstruct_tensor[-1:]
            pad = last.repeat(len(seq) - secstruct_tensor.shape[0])
            secstruct_tensor = torch.cat([secstruct_tensor, pad], dim=0)


            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))

        data = torch_geometric.data.Data(
            x=X_ca, seq=seq, raw_seq=raw_seq, codon=codon, name=name, organ=organ,
            secstruct=secstruct_tensor,
            node_s=node_s, node_v=node_v,
            edge_s=edge_s, edge_v=edge_v,
            edge_index=edge_index, mask=mask, species_id=species_id
        )
        return data
