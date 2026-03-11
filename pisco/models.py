import numpy as np
import torch
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from transformers import AutoTokenizer, AutoModel
import torch.nn.init as init
import pandas as pd
import os
from pisco.data import (
    AA2CODONS,
    LETTER_TO_NUM,
)
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn

class PISCO_Config(PretrainedConfig):
    # 用于 AutoModel 自动识别
    model_type = "PISCO_Model" 
    
    def __init__(self, node_in_dim=None, node_h_dim=None, edge_in_dim=None, edge_h_dim=None,
                 num_layers=3, drop_rate=0.1, 
                 emb_pro_dim=32, emb_cod_dim=32, emb_spe_dim=32, emb_secstruct_dim=2, num_species=165, 
                 use_species_distribution=False, use_esm=True,
                 esm_model_name="facebook/esm2_t12_35M_UR50D", **kwargs):
        super().__init__(**kwargs)
        self.node_in_dim = node_in_dim   
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim  
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.emb_pro_dim = emb_pro_dim
        self.emb_cod_dim = emb_cod_dim
        self.emb_spe_dim = emb_spe_dim
        self.emb_secstruct_dim = emb_secstruct_dim
        self.num_species = num_species
        self.use_species_distribution = use_species_distribution
        self.use_esm = use_esm
        self.esm_model_name = esm_model_name

        
class PISCO_Model(PreTrainedModel):
    # 必须指定 config_class
    config_class = PISCO_Config 

    def __init__(self, config: PISCO_Config):
        super().__init__(config)

        drop_rate = config.drop_rate
        num_layers = config.num_layers
        esm_model_name = config.esm_model_name

        node_in_dim = config.node_in_dim   
        node_h_dim = config.node_h_dim
        edge_in_dim = config.edge_in_dim
        edge_h_dim = config.edge_h_dim  

        emb_pro_dim = config.emb_pro_dim
        self.use_species_distribution = config.use_species_distribution
        self.emb_secstruct_dim = config.emb_secstruct_dim
        self.num_species = config.num_species # include unknown species
        
        self.emb_spe_dim = config.emb_spe_dim

        self.use_esm = config.use_esm


        # === 原始组件 ===
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        # === 序列 embedding ===
        if self.use_esm:
            self.W_s = nn.Embedding(20, emb_pro_dim)
        else:
            self.W_s = nn.Embedding(21, emb_pro_dim)

        # === 物种 embedding ===
        self.W_species = nn.Embedding(self.num_species, self.emb_spe_dim)
        self.species_to_node_s = nn.Linear(self.emb_spe_dim, node_in_dim[0])

        # === 二级结构 embedding ===
        self.W_secstruct = nn.Embedding(3, self.emb_secstruct_dim)  # 0: coil, 1: helix, 2: sheet

        # === ESM 模型 ===
        if self.use_esm:
            self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
            self.esm_model = AutoModel.from_pretrained(esm_model_name)

            # 获取 ESM 模型的输出维度
            self.esm_hidden_dim = self.esm_model.config.hidden_size  # 例如 320, 480 等

            # 只有当 emb_pro_dim 不等于 esm_hidden_dim 时才加投影
            if emb_pro_dim != self.esm_hidden_dim:
                self.W_esm = nn.Sequential(
                    nn.Linear(self.esm_hidden_dim, emb_pro_dim),
                    nn.LayerNorm(emb_pro_dim),  
                    
                )
            else:
                self.W_esm = nn.Sequential(
                    
                    nn.Linear(self.esm_hidden_dim, self.emb_pro_dim),
                    nn.LayerNorm(self.emb_pro_dim),
                )
                init.eye_(self.W_esm[1].weight)
                init.zeros_(self.W_esm[1].bias)

        node_h_dim_enc = (node_h_dim[0] + emb_pro_dim + self.emb_spe_dim + self.emb_secstruct_dim,
                          node_h_dim[1])

        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim_enc, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim_enc, edge_h_dim, drop_rate=drop_rate, autoregressive=True)
            for _ in range(num_layers))

        self.W_out = GVP(node_h_dim_enc, (6, 0), activations=(None, None))

        # === codon 掩码 ===
        self.aa2valid_codon_ids = {
                                    13: [0, 1],                   # F: ['UUU', 'UUC']
                                    10: [0, 1, 2, 3, 4, 5],       # L: ['UUA', ..., 'CUG']
                                    9: [0, 1, 2],                 # I: ['AUU', 'AUC', 'AUA']
                                    12: [0],                      # M: ['AUG']
                                    19: [0, 1, 2, 3],             # V
                                    15: [0, 1, 2, 3, 4, 5],       # S
                                    14: [0, 1, 2, 3],             # P
                                    16: [0, 1, 2, 3],             # T
                                    0: [0, 1, 2, 3],              # A
                                    18: [0, 1],                   # Y
                                    8: [0, 1],                    # H
                                    5: [0, 1],                    # Q
                                    2: [0, 1],                    # N
                                    11: [0, 1],                   # K
                                    3: [0, 1],                    # D
                                    6: [0, 1],                    # E
                                    4: [0, 1],                    # C
                                    1: [0, 1, 2, 3, 4, 5],        # R
                                    7: [0, 1, 2, 3],              # G
                                    17: [0],                      # W
                                    20: [0, 1, 2]                    # _ 
                                }
        valid_table = torch.zeros((21, 6), dtype=torch.bool)
        for aa_id, valid_ids in self.aa2valid_codon_ids.items():
            valid_table[aa_id, torch.tensor(valid_ids, dtype=torch.long)] = True
        self.register_buffer("valid_table", valid_table)
        
        self.register_buffer("species_codon_probs", torch.zeros((self.num_species, 21, 6), dtype=torch.float32))
        self.has_species_codon_probs = False


        self.init_weights()

    def forward(self, h_V, edge_index, h_E, seq, rawseq, species_id, secstruct, batchlist):
        """
        Forward: 同步预测版，不再依赖 codon 输入。
        输入:
        h_V: (node_s, node_v)
        edge_index: (2, E)
        h_E: (edge_s, edge_v)
        seq: (N,)  氨基酸 id
        rawseq: 原始氨基酸序列 (batch)
        species_id: (B,)
        secstruct: (N,)
        batchlist: (N,)
        输出:
        logits: (N, 6) 每个氨基酸位置的 codon 分布
        """
        device = h_V[0].device
        if species_id.dim() == 0:
            species_id = species_id.unsqueeze(0)
        
        # --- 随机置0以增强泛化性 ---
        if self.training:
            drop_prob = 0.05
            mask = torch.rand_like(species_id.float()) < drop_prob
            species_id = species_id.masked_fill(mask, 0)

        # === 节点 / 边初始投影 ===
        h_V = self.W_v(h_V)   # (N, node_h_dim_scalar), (N, node_h_dim_vec, 3)
        h_E = self.W_e(h_E)   # (E, edge_h_dim_scalar), (E, edge_h_dim_vec, 3)

        # === 氨基酸 embedding (ESM 或 lookup) ===
        if self.use_esm:
            inputs = self.esm_tokenizer(
                rawseq, return_tensors="pt", add_special_tokens=False, padding=True
            ).to(device)
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
            esm_hidden = outputs.last_hidden_state    # (B, L, D)
            attention_mask = inputs["attention_mask"] # (B, L)

            B, L, D = esm_hidden.shape
            esm_hidden = esm_hidden.view(B * L, D)
            mask = attention_mask.view(-1).bool()
            esm_hidden = esm_hidden[mask]        # (N, D)
            esm_hidden = self.W_esm(esm_hidden)  # (N, emb_pro_dim)
            h_S = esm_hidden
        else:
            h_S = self.W_s(seq)                  # (N, emb_pro_dim)

        # === 物种 embedding (可选) ===
        if self.emb_spe_dim > 0:
            h_species = self.W_species(species_id)        # (B, emb_species_dim)
            h_species_broadcast = h_species[batchlist]    # (N, emb_species_dim)
        else:
            h_species_broadcast = None


        # === 二级结构 embedding (可选) ===
        if self.emb_secstruct_dim > 0:
            h_SS = self.W_secstruct(secstruct)            # (N, emb_secstruct_dim)
        else:
            h_SS = None


        # === 拼接 ===
        features = [h_V[0], h_S]
        if h_species_broadcast is not None:
            features.append(h_species_broadcast)
        if h_SS is not None:
            features.append(h_SS)

        node_scalar = torch.cat(features, dim=-1)


        h_V = (node_scalar, h_V[1])

        # === Encoder ===
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        encoder_embeddings = h_V

        # === Decoder ===
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)   # (N, 6)

        # === 物种分布约束 (可选) ===
        if self.use_species_distribution:
            if not self.has_species_codon_probs:
                raise RuntimeError("use_species_distribution=True but probs not loaded.")
            species_per_node = species_id[batchlist]      # (N,)
            probs_per_node_all = self.species_codon_probs[species_per_node]  # (N, 20, 6)
            probs_per_node = probs_per_node_all[
                torch.arange(seq.shape[0], device=seq.device), seq.long()
            ]  # (N, 6)
            logits = logits + torch.log(probs_per_node + 1e-9)

        # === Mask 无效 codon ===
        valid_mask_per_pos = self.valid_table[seq]   # (N, 6)
        logits = logits.masked_fill(~valid_mask_per_pos, torch.finfo(logits.dtype).min)
        return logits


    def set_species_codon_probs(self, probs: torch.Tensor):
        """
        probs: tensor shape (probs_num_species, 21, 6) float and already normalized per (species,aa).
        This will copy into the module buffer so it moves with the model.
        """
        if not isinstance(probs, torch.Tensor):
            raise ValueError("probs must be a torch.Tensor")
        if probs.shape != (self.num_species, 21, 6):
            raise ValueError(f"probs must have shape ({self.num_species},21,6), got {probs.shape}")
        # copy into buffer (keeps device semantics)        self.species_codon_probs.data.copy_(probs.to(self.species_codon_probs.device, dtype=self.species_codon_probs.dtype))
        self.has_species_codon_probs = True

    def infer(self, h_V, edge_index, h_E, seq, rawseq, species_id, secstruct,
              species_name=None, csv_path=None):
        """
        Predict codon logits for a single protein (no batchlist).
        Inputs:
            h_V: tuple (node_s, node_v), shape (N, ...)
            edge_index: (2, E)
            h_E: tuple (edge_s, edge_v)
            seq: (N,)  (aa ids)
            rawseq: str or list of str (single sequence)
            species_id: int or tensor (scalar)
            secstruct: (N,)
            species_name: str, optional, species name when species_id==0
            csv_path: str, optional, path to species-codon table CSV
        Returns:
            logits: (N, 6)
        """
        device = h_V[0].device
        if isinstance(species_id, int):
            species_id = torch.tensor([species_id], device=device)
        elif isinstance(species_id, torch.Tensor) and species_id.dim() == 0:
            species_id = species_id.unsqueeze(0)

        # === 节点 / 边初始投影 ===
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # === 氨基酸 embedding (ESM 或 lookup) ===
        if self.use_esm:
            inputs = self.esm_tokenizer(
                rawseq, return_tensors="pt", add_special_tokens=False, padding=True
            ).to(device)
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
            esm_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            L, D = esm_hidden.shape[1], esm_hidden.shape[2]
            esm_hidden = esm_hidden.view(L, D)
            mask = attention_mask.view(-1).bool()
            esm_hidden = esm_hidden[mask]
            esm_hidden = self.W_esm(esm_hidden)
            h_S = esm_hidden
        else:
            h_S = self.W_s(seq)

        # === 物种 embedding ===
        if self.emb_spe_dim > 0:
            h_species = self.W_species(species_id)
            h_species_broadcast = h_species.expand(seq.shape[0], -1)
        else:
            h_species_broadcast = None

        # === 二级结构 embedding ===
        if self.emb_secstruct_dim > 0:
            h_SS = self.W_secstruct(secstruct)
        else:
            h_SS = None

        # === 拼接 ===
        features = [h_V[0], h_S]
        if h_species_broadcast is not None:
            features.append(h_species_broadcast)
        if h_SS is not None:
            features.append(h_SS)
        node_scalar = torch.cat(features, dim=-1)
        h_V = (node_scalar, h_V[1])

        # === Encoder ===
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        encoder_embeddings = h_V

        # === Decoder ===
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)   # (N, 6)
        # === 物种分布约束 ===
        if self.use_species_distribution:
            use_fallback = False

            # 默认使用 species_id 分布
            if species_id.item() != 0:
                use_fallback = True

            # === 尝试从 CSV 动态加载 ===
            elif species_id.item() == 0:
                try:
                    # 基本检查
                    if csv_path is None or species_name is None:
                        raise ValueError("Missing csv_path or species_name")

                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"CSV file not found: {csv_path}")

                    # 初始化缓存
                    if not hasattr(self, "cached_csv_probs"):
                        self.cached_csv_probs = {}

                    # 缓存未命中时加载
                    if species_name not in self.cached_csv_probs:
                        df = pd.read_csv(csv_path)
                        row = df[df["species"] == species_name]
                        if row.empty:
                            raise KeyError(f"Species '{species_name}' not found in {csv_path}")

                        aa_cols = [c for c in df.columns if c not in ("species_id", "species")]
                        values_dict = {col: float(row[col].values[0]) for col in aa_cols}

                        # 初始化全零矩阵 (21, 6)
                        values = torch.zeros((21, 6), dtype=torch.float32)

                        # 按 AA2CODONS 映射填入对应 codon 的值
                        for aa, codons in AA2CODONS.items():
                            aa_idx = LETTER_TO_NUM[aa]
                            for j, codon in enumerate(codons):
                                if codon in values_dict:
                                    values[aa_idx, j] = values_dict[codon]
                                # 若 CSV 没该列，保持 0

                            # 若有 0 值则整行 +1
                            mask_zero = (values == 0).any(dim=1)
                            values[mask_zero] += 1.0

                            # 归一化
                            probs = values / values.sum(dim=1, keepdim=True)
                            self.cached_csv_probs[species_name] = probs.to(device)
                        print(f'load {species_name} from codon_usage_plug')
                    probs_all = self.cached_csv_probs[species_name]
                    probs_per_node = probs_all[seq.long()]  # (N, 6)

                except Exception as e:
                    # 如果发生任何错误（缺文件/物种/参数），自动回退
                    print(f"[WARN] Failed to load species probs from CSV ({e}); using default species_id branch.")
                    use_fallback = True

            # === species_id 分支（默认逻辑或回退） ===
            if use_fallback:
                if not self.has_species_codon_probs:
                    raise RuntimeError("use_species_distribution=True but probs not loaded.")
                species_per_node = species_id.expand(seq.shape[0])
                probs_per_node_all = self.species_codon_probs[species_per_node]  # (N, 21, 6)
                probs_per_node = probs_per_node_all[
                    torch.arange(seq.shape[0], device=seq.device), seq.long()
                ]  # (N, 6)

            logits = logits + torch.log(probs_per_node + 1e-9)


        # === Mask 无效 codon ===
        valid_mask_per_pos = self.valid_table[seq]
        logits = logits.masked_fill(~valid_mask_per_pos, torch.finfo(logits.dtype).min)

        return logits




class PISCO_AR_Model(PreTrainedModel):
    # 必须指定 config_class
    config_class = PISCO_Config 

    def __init__(self, config: PISCO_Config):
        super().__init__(config)

        drop_rate = config.drop_rate
        num_layers = config.num_layers
        esm_model_name = config.esm_model_name
        node_in_dim = config.node_in_dim   
        node_h_dim = config.node_h_dim
        edge_in_dim = config.edge_in_dim
        edge_h_dim = config.edge_h_dim  
        emb_pro_dim = config.emb_pro_dim
        self.use_species_distribution = config.use_species_distribution
        self.use_esm = config.use_esm
        self.emb_cod_dim = config.emb_cod_dim
        self.node_in_dim = config.node_in_dim   
        self.edge_in_dim = config.edge_in_dim  
        self.emb_pro_dim = config.emb_pro_dim
        self.emb_spe_dim = config.emb_spe_dim
        self.emb_secstruct_dim = config.emb_secstruct_dim
        self.num_species = config.num_species
        # === 原始组件 ===
        self.W_v = nn.Sequential(
            GVP(self.node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(self.edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        # === 序列 & codon embedding ===
        self.W_s = nn.Embedding(21, self.emb_pro_dim)
        self.W_codon = nn.Embedding(6, self.emb_cod_dim)

        # === 物种 embedding ===
        self.W_species = nn.Embedding(self.num_species, self.emb_spe_dim)
        self.species_to_node_s = nn.Linear(self.emb_spe_dim, self.node_in_dim[0])

        # === 二级结构 embedding ===
        self.W_secstruct = nn.Embedding(3, config.emb_secstruct_dim) # 0: coil, 1: helix, 2: sheet

        # === ESM 模型 ===
        if self.use_esm:
            self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
            self.esm_model = AutoModel.from_pretrained(esm_model_name)

            # 获取 ESM 模型的输出维度
            self.esm_hidden_dim = self.esm_model.config.hidden_size  # 例如 320, 480 等

            # 只有当 emb_pro_dim 不等于 esm_hidden_dim 时才加投影
            if self.emb_pro_dim != self.esm_hidden_dim:
                # self.W_esm = nn.Sequential(
                #     nn.LayerNorm(self.esm_hidden_dim),   # 避免爆炸
                #     nn.Linear(self.esm_hidden_dim, self.emb_pro_dim)
                # )
                self.W_esm = nn.Sequential(
                    nn.Linear(self.esm_hidden_dim, self.emb_pro_dim),
                    nn.LayerNorm(self.emb_pro_dim),   # 避免爆炸
                    
                )
                # init.xavier_uniform_(self.W_esm[1].weight)
                # init.zeros_(self.W_esm[1].bias)
            else:
                self.W_esm = nn.Sequential(
                    nn.LayerNorm(self.esm_hidden_dim),
                    nn.Linear(self.esm_hidden_dim, self.emb_pro_dim)
                )
                # 初始化成 identity
                init.eye_(self.W_esm[1].weight)
                init.zeros_(self.W_esm[1].bias)

        if self.emb_secstruct_dim > 0:
            node_h_dim_enc = (node_h_dim[0] + self.emb_pro_dim + self.emb_spe_dim + self.emb_secstruct_dim,
                          node_h_dim[1])
        else:
            node_h_dim_enc = (node_h_dim[0] + self.emb_pro_dim + self.emb_spe_dim,
                          node_h_dim[1])

        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim_enc, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        edge_h_dim = (edge_h_dim[0] + self.emb_cod_dim, edge_h_dim[1])



        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim_enc, edge_h_dim, drop_rate=drop_rate, autoregressive=True)
            # GVPConvLayer(node_h_dim_enc, edge_h_dim, drop_rate=drop_rate, autoregressive=True)
            for _ in range(num_layers))

        self.W_out = GVP(node_h_dim_enc, (6, 0), activations=(None, None))
        # self.W_out = GVP(node_h_dim_enc, (6, 0), activations=(None, None))

        # === codon 掩码 ===
        self.aa2valid_codon_ids = {
                                    13: [0, 1],                     # F: ['UUU', 'UUC']
                                    10: [0, 1, 2, 3, 4, 5],         # L: ['UUA', ..., 'CUG']
                                    9: [0, 1, 2],                 # I: ['AUU', 'AUC', 'AUA']
                                    12: [0],                       # M: ['AUG']
                                    19: [0, 1, 2, 3],              # V
                                    15: [0, 1, 2, 3, 4, 5],        # S
                                    14: [0, 1, 2, 3],              # P
                                    16: [0, 1, 2, 3],              # T
                                    0: [0, 1, 2, 3],              # A
                                    18: [0, 1],                    # Y
                                    8: [0, 1],                    # H
                                    5: [0, 1],                    # Q
                                    2: [0, 1],                    # N
                                    11: [0, 1],                    # K
                                    3: [0, 1],                    # D
                                    6: [0, 1],                    # E
                                    4: [0, 1],                    # C
                                    1: [0, 1, 2, 3, 4, 5],        # R
                                    7: [0, 1, 2, 3],              # G
                                    17: [0],                       # W
                                }
        valid_table = torch.zeros((21, 6), dtype=torch.bool)
        for aa_id, valid_ids in self.aa2valid_codon_ids.items():
            valid_table[aa_id, torch.tensor(valid_ids, dtype=torch.long)] = True
        self.register_buffer("valid_table", valid_table)

        self.register_buffer("species_codon_probs", torch.zeros((self.num_species, 21, 6), dtype=torch.float32))
        self.has_species_codon_probs = False
        print('~~~ sd_alpha=0.1 ~~~')

    def forward(self, h_V, edge_index, h_E, seq, rawseq, codon, species_id, secstruct, batchlist):
        """
        Forward that supports both use_species_token==False and ==True.
        Inputs (before augmentation):
        h_V: tuple (node_s, node_v)
            node_s: (N, node_in_dim[0])
            node_v: (N, node_in_dim[1], 3)
        edge_index: (2, E)
        h_E: tuple (edge_s, edge_v)
            edge_s: (E, edge_in_dim[0])
            edge_v: (E, edge_in_dim[1], 3)
        seq: (N,)  (aa ids)
        codon: (N,) (codon indices)
        species_id: (B,)  (per-graph species ids)
        batchlist: (N,) mapping node -> graph index in [0, B-1]
        Returns:
        logits: masked logits aligned to nodes (if token mode, contains species nodes too; you clamp later)
        """
        device = h_V[0].device
        # If species_id is scalar for single graph, make it (1,)
        if species_id.dim() == 0:
            species_id = species_id.unsqueeze(0)

        # --- 随机置0以增强泛化性 ---
        if self.training:
            drop_prob = 0.05
            mask = torch.rand_like(species_id.float()) < drop_prob
            species_id = species_id.masked_fill(mask, 0)

        # apply original W_v / W_e first
        h_V = self.W_v(h_V)                                   # (N, node_h_dim_scalar),(N, node_h_dim_vec,3)
        h_E = self.W_e(h_E)                                   # (E, edge_h_dim_scalar),(E, edge_h_dim_vec,3)

        # === ESM 序列 embedding ===
        if self.use_esm:
            inputs = self.esm_tokenizer(rawseq, return_tensors="pt", add_special_tokens=False, padding = True).to(device)
            with torch.no_grad():
                outputs = self.esm_model(**inputs)

            esm_hidden = outputs.last_hidden_state # (B, L, D)
            attention_mask = inputs["attention_mask"] # (B, L)
            
            B, L, D = esm_hidden.shape
            esm_hidden = esm_hidden.view(B * L, D)
            mask = attention_mask.view(-1).bool()

            # 过滤掉 padding
            esm_hidden = esm_hidden[mask]        # (N, D)
            esm_hidden = self.W_esm(esm_hidden)  # (N, emb_pro_dim)
            h_S = esm_hidden                     # (N, emb_pro_dim)
        else:
            h_S = self.W_s(seq)                  # (N, emb_pro_dim)


        # === species embedding ===
        if self.emb_spe_dim > 0:
            h_species = self.W_species(species_id)        # (B, emb_species_dim)
            h_species_broadcast = h_species[batchlist]    # (N, emb_species_dim)
        else:
            h_species_broadcast = None

        # === secondary structure embedding ===
        if self.emb_secstruct_dim > 0:
            h_SS = self.W_secstruct(secstruct)                    # (N, emb_secstruct_dim)
        else:
            h_SS = None

        # === 拼接 ===
        features = [h_V[0], h_S]
        if h_species_broadcast is not None:
            features.append(h_species_broadcast)
        if h_SS is not None:
            features.append(h_SS)

        node_scalar = torch.cat(features, dim=-1)

        h_V = (node_scalar, h_V[1])

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        encoder_embeddings = h_V

        # decoder: 
        h_C = self.W_codon(codon)                             # (N, emb_codon_dim)
        h_C = h_C[edge_index[0]]                              # (E, emb_codon_dim)
        valid_edge_mask = (edge_index[0] < edge_index[1]).view(-1, 1)
        h_C = h_C * valid_edge_mask.to(h_C.dtype)             # zero out invalid directions
        h_E = (torch.cat([h_E[0], h_C], dim=-1), h_E[1])

        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)

        # === 物种分布约束 (可选) ===
        if self.use_species_distribution:
            if not self.has_species_codon_probs:
                raise RuntimeError("use_species_distribution=True but probs not loaded.")
            species_per_node = species_id[batchlist]      # (N,)
            probs_per_node_all = self.species_codon_probs[species_per_node]  # (N, 21, 6)
            probs_per_node = probs_per_node_all[
                torch.arange(seq.shape[0], device=seq.device), seq.long()
            ]  # (N, 6)
            logits = logits + 0.1 * torch.log(probs_per_node + 1e-9)

        # mask invalid codons
        valid_mask_per_pos = self.valid_table[seq]   # (N,6)
        logits = logits.masked_fill(~valid_mask_per_pos, torch.finfo(logits.dtype).min)
        return logits


    def set_species_codon_probs(self, probs: torch.Tensor):
        """
        probs: tensor shape (num_species, 20, 6) float and already normalized per (species,aa).
        This will copy into the module buffer so it moves with the model.
        """
        if not isinstance(probs, torch.Tensor):
            raise ValueError("probs must be a torch.Tensor")
        if probs.shape != (self.num_species, 21, 6):
            raise ValueError(f"probs must have shape ({self.num_species},21,6), got {probs.shape}")
        # copy into buffer (keeps device semantics)
        self.species_codon_probs.data.copy_(probs.to(self.species_codon_probs.device, dtype=self.species_codon_probs.dtype))
        self.has_species_codon_probs = True

    def sample(self, h_V, edge_index, h_E, seq, raw_seq, species_id, secstruct):
        return self.deterministic_sample(h_V, edge_index, h_E, seq, raw_seq, species_id, secstruct)

    def infer(self, h_V, edge_index, h_E, seq, raw_seq, species_id, secstruct,
              species_name=None, csv_path=None):
        return self.deterministic_sample(h_V, edge_index, h_E, seq, raw_seq, species_id, secstruct,
          return_logits=True,species_name=species_name, csv_path=csv_path)

    def deterministic_sample(
        self,
        h_V, edge_index, h_E,
        seq, rawseq,
        species_id, secstruct,
        return_logits: bool = False,
        species_name: str = None,
        csv_path: str = None,
    ):
        """
        species_id: scalar tensor, 表示物种
        species_name + csv_path: 当 species_id==0 时才会尝试从 CSV 动态加载
        """
        with torch.inference_mode():
            device = edge_index.device
            L = h_V[0].shape[0]

            if return_logits:
                logits_all = torch.zeros(
                    L, 6, device=device, dtype=h_V[0].dtype
                )

            # === encoder ===
            h_V = self.W_v(h_V)
            h_E = self.W_e(h_E)

            # === First embeddings ===
            if self.use_esm:
                inputs = self.esm_tokenizer(
                    rawseq, return_tensors="pt", add_special_tokens=False, padding=True
                ).to(device)
                with torch.no_grad():
                    outputs = self.esm_model(**inputs)

                esm_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                B, L2, D = esm_hidden.shape
                esm_hidden = esm_hidden.view(B * L2, D)
                mask = attention_mask.view(-1).bool()
                esm_hidden = esm_hidden[mask]
                esm_hidden = self.W_esm(esm_hidden)
                h_S = esm_hidden
            else:
                h_S = self.W_s(seq)  # (N, emb_pro_dim)

            # === species embedding ===
            if self.emb_spe_dim > 0:
                h_species = self.W_species(species_id)  # (1, emb_species_dim)
                h_species_broadcast = h_species.expand(L, -1)
            else:
                h_species_broadcast = None

            # === secondary structure embedding ===
            if self.emb_secstruct_dim > 0:
                h_SS = self.W_secstruct(secstruct)
            else:
                h_SS = None

            # === concat ===
            features = [h_V[0], h_S]
            if h_species_broadcast is not None:
                features.append(h_species_broadcast)
            if h_SS is not None:
                features.append(h_SS)

            node_scalar = torch.cat(features, dim=-1)
            h_V = (node_scalar, h_V[1])

            # === Encoder Layers ===
            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)
            h_V_enc = h_V

            # -------------------------------------------------
            # === Species distribution preparation (CSV logic) ===
            # -------------------------------------------------
            mode = "builtin"
            csv_probs = None

            if self.use_species_distribution:
                sid = species_id.item() if isinstance(species_id, torch.Tensor) else int(species_id)

                if sid != 0:
                    # 普通情况：直接用 builtin species_codon_probs
                    if not self.has_species_codon_probs:
                        raise RuntimeError("species distribution enabled but builtin probs not loaded")
                    mode = "builtin"

                else:
                    # sid == 0 → 尝试 CSV 加载
                    try:
                        if csv_path is None or species_name is None:
                            raise ValueError("Missing csv_path or species_name for sid=0")

                        if not os.path.exists(csv_path):
                            raise FileNotFoundError(f"CSV file not found: {csv_path}")

                        # initialize cache
                        if not hasattr(self, "cached_csv_probs"):
                            self.cached_csv_probs = {}

                        # cache miss → load CSV
                        if species_name not in self.cached_csv_probs:
                            df = pd.read_csv(csv_path)

                            row = df[df["species"] == species_name]
                            if row.empty:
                                raise KeyError(f"Species '{species_name}' not found in CSV.")

                            aa_cols = [c for c in df.columns if c not in ("species_id", "species")]
                            values_dict = {col: float(row[col].values[0]) for col in aa_cols}

                            # build (21, 6)
                            values = torch.zeros((21, 6), dtype=torch.float32)

                            for aa, codons in AA2CODONS.items():
                                aa_idx = LETTER_TO_NUM[aa]
                                for j, codon in enumerate(codons):
                                    if codon in values_dict:
                                        values[aa_idx, j] = values_dict[codon]

                            # 对任何出现0的行 +1
                            mask_zero = (values == 0).any(dim=1)
                            values[mask_zero] += 1.0

                            # normalize
                            probs = values / values.sum(dim=1, keepdim=True)
                            self.cached_csv_probs[species_name] = probs.to(device)

                        csv_probs = self.cached_csv_probs[species_name]  # (21, 6)
                        mode = "csv"

                    except Exception as e:
                        print(f"[WARN] CSV species distribution failed: {e}")
                        print("[WARN] Falling back to builtin species distribution.")
                        if not self.has_species_codon_probs:
                            raise RuntimeError("fallback failed: builtin species probs not loaded")
                        mode = "builtin"

            # -------------------------------------------------
            # === Decoder (autoregressive) ===
            # -------------------------------------------------
            codon_single = torch.zeros(L, dtype=torch.long, device=device)
            h_C = torch.zeros(L, self.emb_cod_dim, device=device)
            h_V_cache = [(h_V_enc[0].clone(), h_V_enc[1].clone()) for _ in self.decoder_layers]

            for i in range(L):
                h_C_edge = h_C[edge_index[0]]
                h_C_edge[edge_index[0] >= edge_index[1]] = 0
                h_E_ = (torch.cat([h_E[0], h_C_edge], dim=-1), h_E[1])

                edge_mask = (edge_index[1] == i)
                edge_index_ = edge_index[:, edge_mask]
                h_E_masked = tuple_index(h_E_, edge_mask)

                node_mask = torch.zeros(L, dtype=torch.bool, device=device)
                node_mask[i] = True

                out = None
                for j, layer in enumerate(self.decoder_layers):
                    out_j = layer(
                        h_V_cache[j],
                        edge_index_,
                        h_E_masked,
                        autoregressive_x=h_V_cache[0],
                        node_mask=node_mask
                    )
                    out_j = tuple_index(out_j, node_mask)

                    if j < len(self.decoder_layers) - 1:
                        h_V_cache[j+1][0][i:i+1] = out_j[0]
                        h_V_cache[j+1][1][i:i+1] = out_j[1]

                    out = out_j

                logits = self.W_out(out).squeeze(0)
                aa_id = seq[i]

                # === apply species distribution ===
                if self.use_species_distribution:
                    if mode == "csv":
                        probs_aa = csv_probs[aa_id]  # (6,)
                    else:
                        probs_aa = self.species_codon_probs[species_id, aa_id]  # (6,)

                    logits = logits + torch.log(probs_aa + 1e-9)

                # === mask invalid codons ===
                valid_mask = self.valid_table[aa_id]
                logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

                best_idx = torch.argmax(logits, dim=-1)
                codon_single[i] = best_idx
                h_C[i] = self.W_codon(best_idx)

                h_V_cache[0][0][i:i+1] = out[0]
                h_V_cache[0][1][i:i+1] = out[1]

                if return_logits:
                    logits_all[i] = logits

            if return_logits:
                return codon_single.unsqueeze(0), logits_all.unsqueeze(0)
            else:
                return codon_single.unsqueeze(0)
