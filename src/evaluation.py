from typing import Dict, List, Tuple
import json
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import math
from collections import Counter,defaultdict
import torch
import torch.nn.functional as F

codon_to_amino_acid = {
    'UUU': 'F', 'UUC': 'F',
    'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    'AUG': 'M',  # start condon
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y',
    # 'UAA': '#', 'UAG': '#', 'UGA': '#',
    'CAU': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C',
    'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S','UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    "UAA": "_", "UAG": "_", "UGA": "_"
}


amino_acid_to_codon = {
    'F': ['UUU', 'UUC'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'], #1
    'I': ['AUU', 'AUC', 'AUA'],
    'M': ['AUG'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'Y': ['UAU', 'UAC'],
    '_': ['UAA', 'UAG', 'UGA'],
    'H': ['CAU', 'CAC'],
    'Q': ['CAA', 'CAG'],
    'N': ['AAU', 'AAC'],
    'K': ['AAA', 'AAG'],
    'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'],
    'C': ['UGU', 'UGC'],
    'W': ['UGG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['AGU', 'AGC', 'UCU', 'UCC', 'UCA', 'UCG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG']
}

def _split_into_codons(seq: str,AA_seq:str):
    """Yield successive 3-letter chunks of a string/sequence."""
    res=''
    for i in range(0, len(seq), 3):
            #if codon_to_amino_acid[seq[i:i + 3]] == 'Stop':
            #     #continue
            #     yield codon_to_amino_acid[seq[i:i + 3]]+seq[i:i + 3]
            # else:
            res+=codon_to_amino_acid[seq[i:i + 3]]
    AA_seq=AA_seq[:len(res)]
    print("generate AA seq",res)
    print("original AA seq",AA_seq)
    return res[:-1]==AA_seq[:-1]

def AA_tokenize(seq: str):
    token=''
    for i in range(len(seq)-1):
        if seq[i]=='L':
           token+=("*U*")
        elif seq[i]=='R':
           token+=("*G*")
        elif seq[i]=='S':
           token+=("***")
        else:
            token+=amino_acid_to_codon[seq[i]][0][:2]
            token+='*'
    token+="U**"
    print("seq",seq)
    print("token",token)
    return token

def calculate_similarity(dna1, dna2):
            """
            Calculate similarity score between two DNA sequences.
            For every three codons (triplets), if they are the same, similarity score increases by 1.

            Args:
                dna1 (str): First DNA sequence.
                dna2 (str): Second DNA sequence.

            Returns:
                int: Similarity score.
            """
            # Ensure both DNA sequences are of the same length
            min_length = min(len(dna1), len(dna2))
            dna1 = dna1[:min_length]
            dna2 = dna2[:min_length]

            # Calculate similarity
            similarity_score = 0
            for i in range(0, min_length, 3):  # Iterate in steps of 3 (codon)
                if dna1[i:i+3] == dna2[i:i+3]:
                    similarity_score += 1

            return similarity_score/(min_length/3)

codon_dict = {
    "ALA": ["GCU", "GCC", "GCA", "GCG"],
    "ARG": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "ASN": ["AAU", "AAC"],
    "ASP": ["GAU", "GAC"],
    "CYS": ["UGU", "UGC"],
    "GLN": ["CAA", "CAG"],
    "GLU": ["GAA", "GAG"],
    "GLY": ["GGU", "GGC", "GGA", "GGG"],
    "HIS": ["CAU", "CAC"],
    "ILE": ["AUU", "AUC", "AUA"],
    "LEU": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "LYS": ["AAA", "AAG"],
    "MET": ["AUG"],
    "PHE": ["UUU", "UUC"],
    "PRO": ["CCU", "CCC", "CCA", "CCG"],
    "SER": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "THR": ["ACU", "ACC", "ACA", "ACG"],
    "TRP": ["UGG"],
    "TYR": ["UAU", "UAC"],
    "VAL": ["GUU", "GUC", "GUA", "GUG"],
}

"""
organism_codon_weights = json.load(open('/ai/share/workspace/wwtan/wzjin/code/gvp-pytorch/data/organ/organism_codon_weights.json', 'r'))
organism_codon_weights_modified = organism_codon_weights.copy()
for organism_i in organism_codon_weights:
    for aa_i in codon_dict:
        count_list_temp = []
        for codon_i in codon_dict[aa_i]:
            count_list_temp.append(organism_codon_weights[organism_i][codon_i])
        for codon_i in codon_dict[aa_i]:
            organism_codon_weights_modified[organism_i][codon_i] = organism_codon_weights[organism_i][codon_i] / max(count_list_temp)

organism_codon_weights = organism_codon_weights_modified
organism_codon_frequency_dict = {}
for organism_i in organism_codon_weights:
    organism_codon_frequency_dict[organism_i] = {}
    for aa_i in codon_dict:
        codon_list_i = []
        weights_list_i = []
        for codon_i in codon_dict[aa_i]:
            codon_list_i.append(codon_i)
            weights_list_i.append(organism_codon_weights[organism_i][codon_i])
        organism_codon_frequency_dict[organism_i][codon_i] = (codon_list_i, weights_list_i)
"""

def convert_codon_usgage_to_relative_weights(codon_usage_dict, include_stop: bool = False):
    """Convert codon usage counts to relative weights per amino acid.
    If include_stop is False (default), entries with amino acid '_' will be skipped.
    """
    codon_relative_weights_dict = {}
    for aa_i in codon_usage_dict:
        # skip stop codons unless user requests to include them
        if aa_i == '_' and not include_stop:
            continue
        counts = list(codon_usage_dict[aa_i][1])
        if len(counts) == 0:
            print(f"Warning: No codons found for amino acid '{aa_i}'. Skipping.")
            continue
        max_count = max(counts)
        # avoid division by zero
        if max_count == 0:
            weights_list_i = [0.0 for _ in counts]
        else:
            weights_list_i = [c / max_count for c in counts]
        codon_list_i = list(codon_usage_dict[aa_i][0])
        codon_relative_weights_dict[aa_i] = (codon_list_i, weights_list_i)
    return codon_relative_weights_dict

def get_cfd(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    threshold: float = 0.3,
) -> float:
    """
    Calculate the codon frequency distribution (CFD) metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        threshold (float): Frequency threshold for counting rare codons.

    Returns:
        float: The CFD metric as a percentage.
    """
    # Get a dictionary mapping each codon to its normalized frequency
    codon2frequency = {
        codon: freq / max(frequencies)
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon, freq in zip(codons, frequencies)
    }
    #print("codon2frequency",codon2frequency)

    cfd = 0

    # Iterate through the DNA sequence in steps of 3 to process each codon
    for i in range(0, len(dna), 3):
        codon = dna[i : i + 3]
        codon_frequency = codon2frequency[codon]

        if codon_frequency < threshold:
            cfd += 1

    return cfd / (len(dna) / 3) * 100

def get_min_max_percentage(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    window_size: int = 18,
) -> List[float]:
    """
    Calculate the %MinMax metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
            frequency distribution per amino acid.
        window_size (int): Size of the window to calculate %MinMax.

    Returns:
        List[float]: List of %MinMax values for the sequence.

    Credit: https://github.com/chowington/minmax
    """
    # Get a dictionary mapping each codon to its respective amino acid
    codon2amino = {
        codon: amino
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon in codons
    }

    min_max_values = []
    codons = [dna[i : i + 3] for i in range(0, len(dna), 3)]
    
    # Iterate through the DNA sequence using the specified window size
    for i in range(len(codons) - window_size + 1):
        codon_window = codons[i : i + window_size]  # Codons in the current window

        Actual = 0.0  # Average of the actual codon frequencies
        Max = 0.0  # Average of the min codon frequencies
        Min = 0.0  # Average of the max codon frequencies
        Avg = 0.0  # Average of the averages of all frequencies for each amino acid

        # Sum the frequencies for codons in the current window
        for codon in codon_window:
            aminoacid = codon2amino[codon]
            frequencies = codon_frequencies[aminoacid][1]
            codon_index = codon_frequencies[aminoacid][0].index(codon)
            codon_frequency = codon_frequencies[aminoacid][1][codon_index]

            Actual += codon_frequency
            Max += max(frequencies)
            Min += min(frequencies)
            Avg += sum(frequencies) / len(frequencies)

        # Divide by the window size to get the averages
        Actual = Actual / window_size
        Max = Max / window_size
        Min = Min / window_size
        Avg = Avg / window_size

        # Calculate %MinMax
        percentMax = ((Actual - Avg) / (Max - Avg)) * 100
        percentMin = ((Avg - Actual) / (Avg - Min)) * 100

        # Append the appropriate %MinMax value
        if percentMax >= 0:
            min_max_values.append(percentMax)
        else:
            min_max_values.append(-percentMin)

    # Populate the last floor(window_size / 2) entries of min_max_values with None
    for i in range(int(window_size / 2)):
        min_max_values.append(None)

    return min_max_values

def get_dtw(seq1, seq2):
    seq1_clean = [[x/100] for x in seq1 if x is not None]
    seq2_clean = [[x/100] for x in seq2 if x is not None]
    seq1_clean = np.array(seq1_clean)
    seq2_clean = np.array(seq2_clean)
    distance, _ = fastdtw(seq1_clean, seq2_clean, dist=euclidean)
    normalized_distance = distance / max(len(seq1_clean), len(seq2_clean))
    return normalized_distance

#calculate KL-divergence from two sets, not two distributions.
def calculate_kl_divergence(data1, data2, bins=10):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    combined_data = np.concatenate([data1, data2])
    min_val = np.min(combined_data)
    max_val = np.max(combined_data)
    hist1, bin_edges = np.histogram(data1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
    hist1 = np.where(hist1 == 0, 1e-10, hist1)
    hist2 = np.where(hist2 == 0, 1e-10, hist2)
    kl_divergence = entropy(hist1, hist2)
    return kl_divergence

def get_gc_percent(seq):
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return float(gc) / len(seq) * 100.0


class CodonUsageLoader:
    def __init__(self, jsonl_path:str ='', csv_path:str=''):
        self.cached_csv_probs = {}
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path

    def load_codon_usage_from_csv(self, organism: str, csv_path: str):
        """
        从 CSV 文件中加载指定物种的密码子使用频率。

        Args:
            organism (str): 物种名称。
            csv_path (str): CSV 文件路径。
        Returns:
            Dict[str, Tuple[List[str], List[int]]]: 密码子使用频率字
        """
        import os
        import pandas as pd
        try:
            # 基本检查
            if csv_path is None or organism is None:
                raise ValueError("Missing csv_path or species_name")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # 初始化缓存
            if not hasattr(self, "cached_csv_probs"):
                self.cached_csv_probs = {}

            # 缓存未命中时加载
            if organism not in self.cached_csv_probs:
                df = pd.read_csv(csv_path)
                row = df[df["species"] == organism]
                if row.empty:
                    raise KeyError(f"Species '{organism}' not found in {csv_path}")
                codon_freqs = {}
                for _, r in row.iterrows():
                    for aa in amino_acid_to_codon.keys():
                        codons = amino_acid_to_codon[aa]
                        freqs = [int(r[codon]) for codon in codons]
                        codon_freqs[aa] = (codons, freqs)
                self.cached_csv_probs[organism] = codon_freqs
            return self.cached_csv_probs[organism]
        except Exception as e:
            print(f"Error loading codon usage for species '{organism}': {e}")
            return None

    def load_all_species_codon_frequencies(self, jsonl_path):
        """
        读取 jsonl 文件中所有物种的密码子频率信息并缓存。
        
        返回结构：
        {
            "Citrobacter werkmanii": {
                "M": (["ATG"], [35]),
                "P": (["CCT", "CCC", "CCA", "CCG"], [7,5,8,27]),
                ...
            },
            "Another species": {
                ...
            }
        }
        """
        species_codon_freqs = {}

        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                species = list(data.keys())[0]  # 每行只有一个物种
                species_data = data[species]

                formatted = {}
                for aa, (codons, freqs) in species_data.items():
                    formatted[aa] = (codons, freqs)

                species_codon_freqs[species] = formatted
        
        return species_codon_freqs
    
    def convert_csv_row_to_json_format(self, csv_path: str, organism: str) -> Dict[str, Dict[str, List[List[str or int]]]] or None:
            """
            根据物种名称从CSV数据文件中查找一行，并将其转换为指定的JSONL格式（使用 Pandas）。

            :param csv_path: 包含物种数据的CSV文件路径。
            :param organism: 要查找的物种名称（例如："Serratia fonticola"）。
            :return: 转换后的JSON格式字典，如果未找到则返回 None。
            """
            import pandas as pd
            import os
            try:
                if not os.path.exists(csv_path):
                    # 在实际运行时，请确保文件路径正确
                    print(f"Error: CSV file not found at {csv_path}")
                    return None

                # 1. 使用 Pandas 读取 CSV 文件
                df = pd.read_csv(csv_path)
                
                # 2. 查找目标物种的行
                row_data = df[df["species"] == organism]

                if row_data.empty:
                    print(f"Species '{organism}' not found in CSV data.")
                    return None

                # 假设每个物种只有一行数据，提取为 Series
                r = row_data.iloc[0]

                # 3. 初始化最终的 JSON 结构
                result_json = {organism: {}}
                species_data = result_json[organism]

                # 4. 遍历所有密码子，将它们归类到相应的氨基酸下
                # 从第3列（索引 2）开始遍历，因为前两列是 species_id 和 species
                # 也可以直接遍历 codon_to_amino_acid 的键，确保只处理密码子列
                
                # 获取所有密码子列名（排除前两列）
                codon_columns = [col for col in df.columns if col in codon_to_amino_acid]

                for codon in codon_columns:
                    amino_acid = codon_to_amino_acid[codon]
                    
                    # 尝试获取计数并转换为整数
                    try:
                        # 使用 .loc[] 确保访问正确的值
                        count = r.loc[codon] 
                        count = int(float(count)) # CSV中的数据可能是浮点数，先转float再转int
                    except (KeyError, ValueError, TypeError):
                        continue # 如果列不存在或值无效，则跳过
                    
                    # 初始化氨基酸的数据结构: [[codons], [counts]]
                    if amino_acid not in species_data:
                        species_data[amino_acid] = [[], []]

                    # 添加密码子和对应的计数值
                    species_data[amino_acid][0].append(codon)
                    species_data[amino_acid][1].append(count)

                return result_json

            except Exception as e:
                print(f"An error occurred during conversion for species '{organism}': {e}")
                return None

def get_cousin(rna, codon_frequencies):
    """
    计算 COUSIN 指标
    Args:
        rna (str): RNA 序列 (长度为 3 的倍数)
        codon_frequencies: {aa: ([codons], [freqs])}，参考宿主的密码子使用频率表
    Returns:
        float: COUSIN 值 (0~1)
    """
    # 1. 建立 codon→aa 映射
    codon2aa = {codon: aa for aa, (codons, _) in codon_frequencies.items() for codon in codons}
    
    # 2. 序列中每个 codon 的统计
    codon_counts = Counter([rna[i:i+3] for i in range(0, len(rna), 3) if rna[i:i+3] in codon2aa])
    
    # 3. 对每个氨基酸分别计算相似度
    similarities = []
    for aa, (codons, ref_freqs) in codon_frequencies.items():
        total_seq = sum(codon_counts[c] for c in codons)
        total_ref = sum(ref_freqs)
        if total_seq == 0 or total_ref == 0:
            continue
        
        # 序列和参考的分布
        P_seq = [codon_counts[c]/total_seq if total_seq > 0 else 0 for c in codons]
        P_ref = [f/total_ref for f in ref_freqs]
        
        # 计算余弦相似度
        dot = sum(p*q for p, q in zip(P_seq, P_ref))
        norm_seq = math.sqrt(sum(p*p for p in P_seq))
        norm_ref = math.sqrt(sum(q*q for q in P_ref))
        if norm_seq == 0 or norm_ref == 0:
            continue
        
        cos_sim = dot / (norm_seq * norm_ref)
        similarities.append(cos_sim)
    
    if len(similarities) == 0:
        return 0.0
    
    return sum(similarities) / len(similarities)


def get_calculate_similarity(rna_seq, codon_usage):
    """
    计算给定序列的 Codon Similarity Index (CSI)

    参数：
      rna_seq: str，例如 "AUGGCUGCACUA"
      codon_usage: dict，例如 {"A": (["GCU","GCC","GCA","GCG"], [22,42,21,39])}

    返回：
      float — CSI 值 (0~1)
    """
    # 划分三联体
    codons = [rna_seq[i:i+3] for i in range(0, len(rna_seq), 3) if len(rna_seq[i:i+3]) == 3]
    
    # 统计序列中每个氨基酸对应密码子的使用情况
    seq_count = defaultdict(lambda: defaultdict(int))
    for codon in codons:
        aa = codon_to_amino_acid.get(codon)
        if aa in codon_usage:
            seq_count[aa][codon] += 1

    csi_sum, aa_count = 0, 0

    for aa, codons_ref in codon_usage.items():
        codon_list, freq_list = codons_ref
        total_ref = sum(freq_list)
        if total_ref == 0:
            continue
        p_ref = [f / total_ref for f in freq_list]

        total_seq = sum(seq_count[aa].values())
        if total_seq == 0:
            continue
        p_seq = [seq_count[aa].get(c, 0) / total_seq for c in codon_list]

        numerator = sum(p_s * p_r for p_s, p_r in zip(p_seq, p_ref))
        denom_ref = math.sqrt(sum(p_r**2 for p_r in p_ref))

        csi_sum += numerator / denom_ref
        aa_count += 1

    if aa_count == 0:
        return 0.0
    return csi_sum / aa_count

def score_codon_sequence_with_logits(logits, codon_indices, reduction='confidence'):
    """
    logits: Tensor shape [L, C] (L positions, C codon classes)
    codon_indices: Tensor or 1D array of length L with codon ids (int)
    reduction: 'sum' | 'mean' | 'perplexity' | 'confidence' | 'none'
    returns python float (or tensor if reduction=='none')
    """
    if not torch.is_tensor(codon_indices):
        codon_indices = torch.tensor(codon_indices, device=logits.device)
    else:
        codon_indices = codon_indices.to(logits.device).long()
    log_probs = F.log_softmax(logits, dim=-1)               # [L, C]
    pos_idx = torch.arange(logits.size(0), device=logits.device)
    selected = log_probs[pos_idx, codon_indices]           # [L]
    if reduction == 'sum':
        return float(selected.sum().item())
    if reduction == 'mean':
        return float(selected.mean().item())
    if reduction == 'perplexity':
        return float(torch.exp(-selected.mean()).item())
    if reduction == 'confidence':
        return float(torch.exp(selected.mean()).item())
    return selected  # return tensor of per-position log-probs

def get_calculate_similarity_from_frequencies(rna_seq: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> float:
    """
    Compute Codon Similarity Index (CSI) for rna_seq using codon_frequencies in the
    same format as get_min_max_percentage/get_cfd: {aa: ([codons], [freqs])}.
    Frequencies may be counts or relative weights; they will be normalized per amino acid.
    Returns a float in [0,1].
    """
    # Split into codons
    codons = [rna_seq[i:i+3] for i in range(0, len(rna_seq), 3) if len(rna_seq[i:i+3]) == 3]
    # Count observed codons per amino acid
    seq_count = defaultdict(lambda: defaultdict(int))
    for codon in codons:
        aa = codon_to_amino_acid.get(codon)
        if aa in codon_frequencies:
            seq_count[aa][codon] += 1

    csi_sum = 0.0
    aa_count = 0

    for aa, (codon_list, freq_list) in codon_frequencies.items():
        total_ref = sum(freq_list)
        if total_ref == 0:
            continue
        # normalized reference distribution
        p_ref = [f / total_ref for f in freq_list]

        total_seq = sum(seq_count[aa].values())
        if total_seq == 0:
            continue
        # observed distribution for this aa in the sequence
        p_seq = [seq_count[aa].get(c, 0) / total_seq for c in codon_list]

        # cosine similarity
        dot = sum(p_s * p_r for p_s, p_r in zip(p_seq, p_ref))
        norm_seq = math.sqrt(sum(p*p for p in p_seq))
        norm_ref = math.sqrt(sum(q*q for q in p_ref))
        if norm_seq == 0 or norm_ref == 0:
            continue
        cos_sim = dot / (norm_seq * norm_ref)
        csi_sum += cos_sim
        aa_count += 1

    if aa_count == 0:
        return 0.0
    return csi_sum / aa_count

def get_calculate_csi(
    rna_seq: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
) -> float:
    """
    Compute Codon Similarity Index (CSI) as defined in the paper.

    Args:
        rna_seq: RNA sequence (string, length multiple of 3 recommended)
        codon_frequencies: {aa: ([codons], [freqs])}
            freqs can be counts or frequencies; only relative values matter.

    Returns:
        CSI value in [0, 1]
    """
    # Split into codons
    codons = [
        rna_seq[i:i+3]
        for i in range(0, len(rna_seq), 3)
        if len(rna_seq[i:i+3]) == 3
    ]

    log_sum = 0.0
    L = 0

    for codon in codons:
        aa = codon_to_amino_acid.get(codon)
        if aa not in codon_frequencies:
            continue

        codon_list, freq_list = codon_frequencies[aa]
        if codon not in codon_list:
            continue

        x_ij = freq_list[codon_list.index(codon)]
        x_max = max(freq_list)
        if x_ij <= 0 or x_max <= 0:
            continue

        w = x_ij / x_max
        log_sum += math.log(w)
        L += 1

    if L == 0:
        return 0.0

    return math.exp(log_sum / L)


def calculate_codon_accuracy(label: str, prediction: str) -> float:
    """
    计算预测的密码子准确率。通过比较标签（label）和预测（prediction）RNA或DNA序列，
    按照密码子（每3个碱基）逐个检查是否匹配，计算匹配的比例。

    参数：
        label (str): 标签序列（真实的RNA或DNA序列）。
        prediction (str): 预测序列（模型输出的RNA或DNA序列）。

    返回：
        float: 准确率，范围为 [0, 1]，表示密码子匹配正确的比例。
    """
    # 确保输入的序列长度是3的倍数
    if len(label) % 3 != 0 or len(prediction) % 3 != 0:
        raise ValueError("输入序列长度必须是3的倍数")

    # 将标签和预测序列拆分成密码子（3个碱基为一个密码子）
    label_codons = [label[i:i + 3] for i in range(0, len(label), 3)]
    prediction_codons = [prediction[i:i + 3] for i in range(0, len(prediction), 3)]

    # 确保拆分后的密码子数目相同
    if len(label_codons) != len(prediction_codons):
        raise ValueError("标签序列和预测序列的密码子数目不相等")

    # 计算密码子匹配的数量
    correct_count = sum(1 for i in range(len(label_codons)) if label_codons[i] == prediction_codons[i])

    # 计算准确率
    accuracy = correct_count / len(label_codons)
    return accuracy
