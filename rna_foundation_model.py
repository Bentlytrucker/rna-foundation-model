"""
RNA Foundation Model - Physics-Masked Cross-Attention

핵심 설계 원칙:
1. 불변 물리 특징 vs 학습 가능 임베딩 분리
2. 물리 마스킹: H-bond/Stacking 가능 쌍만 attention
3. SE(3) Rigid Transform으로 3D 구조 예측
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import IntEnum
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


# ==============================================================================
# 1. 물리 상수 정의 (불변)
# ==============================================================================

ATOM_PHYSICS = {
    1:  {'symbol': 'H',  'vdw_radius': 1.20, 'mass': 1.008,  'electronegativity': 2.20},
    6:  {'symbol': 'C',  'vdw_radius': 1.70, 'mass': 12.011, 'electronegativity': 2.55},
    7:  {'symbol': 'N',  'vdw_radius': 1.55, 'mass': 14.007, 'electronegativity': 3.04},
    8:  {'symbol': 'O',  'vdw_radius': 1.52, 'mass': 15.999, 'electronegativity': 3.44},
    15: {'symbol': 'P',  'vdw_radius': 1.80, 'mass': 30.974, 'electronegativity': 2.19},
}

HBOND_GEOMETRY = {
    'distance_range': (2.4, 3.5),
    'optimal_distance': 2.8,
}

BASE_SMILES = {
    'A': 'Nc1ncnc2[nH]cnc12',
    'G': 'Nc1nc2[nH]cnc2c(=O)[nH]1',
    'U': 'O=c1cc[nH]c(=O)[nH]1',
    'C': 'Nc1cc[nH]c(=O)n1',
}


class SugarPucker(IntEnum):
    C2_ENDO = 0
    C3_ENDO = 1


def vdw_volume(r):
    return (4/3) * math.pi * (r ** 3)


def get_initial_template(base):
    mol = Chem.MolFromSmiles(BASE_SMILES[base])
    try:
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        return torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, 0.0]
                             for i in range(mol.GetNumAtoms())], dtype=torch.float)
    except:
        n = mol.GetNumAtoms()
        angles = torch.linspace(0, 2*math.pi, n+1)[:-1]
        return torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros(n)], dim=1) * 1.5


# ==============================================================================
# 2. 원자 특징 분류
# ==============================================================================

INVARIANT_PHYSICS_DIM = 10  # 불변 물리 특징 차원
STRUCTURAL_DIM = 4          # 구조적 특징 차원
TOTAL_ATOM_DIM = INVARIANT_PHYSICS_DIM + STRUCTURAL_DIM  # 14


def get_atom_features(atom):
    """
    원자 특징 추출

    Returns:
        physics_features: [10] 불변 물리 특징
        structural_features: [4] 구조적 특징
    """
    atomic_num = atom.GetAtomicNum()
    props = ATOM_PHYSICS.get(atomic_num, ATOM_PHYSICS[6])

    num_hs = atom.GetTotalNumHs()
    is_electronegative = atomic_num in [7, 8]  # N, O

    # 불변 물리 특징 (10-dim) - 절대 학습되지 않음
    physics_features = [
        float(atomic_num),                          # 0: 원자번호
        props['mass'],                              # 1: 질량
        props['vdw_radius'],                        # 2: VdW 반경
        vdw_volume(props['vdw_radius']),            # 3: VdW 부피
        props['electronegativity'],                 # 4: 전기음성도
        float(is_electronegative),                  # 5: 전기음성 원자 여부
        float(is_electronegative and num_hs > 0),   # 6: H-bond donor
        float(is_electronegative),                  # 7: H-bond acceptor
        float(num_hs),                              # 8: 수소 개수
        float(atom.GetFormalCharge()),              # 9: 형식 전하
    ]

    # 구조적 특징 (4-dim) - 분자 구조에서 파생
    hyb = {Chem.rdchem.HybridizationType.SP: 1, Chem.rdchem.HybridizationType.SP2: 2,
           Chem.rdchem.HybridizationType.SP3: 3}.get(atom.GetHybridization(), 0)

    structural_features = [
        float(hyb),                     # 0: 혼성 오비탈
        float(atom.GetIsAromatic()),    # 1: 방향족성
        float(atom.IsInRing()),         # 2: 고리 포함
        float(atom.GetDegree()),        # 3: 결합 차수
    ]

    return physics_features, structural_features


def get_sugar_features(pucker=SugarPucker.C3_ENDO):
    """당 특징 (8차원)"""
    return [1.0, float(pucker == SugarPucker.C2_ENDO), float(pucker == SugarPucker.C3_ENDO),
            1.0, 134.13, 5*vdw_volume(1.70)+4*vdw_volume(1.52), 5.0, 4.0]


def get_phosphate_features():
    """인산 특징 (8차원)"""
    return [2.0, -1.0, 94.97, vdw_volume(1.80)+4*vdw_volume(1.52), 1.0, 4.0, 1.80, 2.19]


# ==============================================================================
# 3. 이질적 RNA 그래프 (물리/구조 분리)
# ==============================================================================

class HeterogeneousRNAGraph:
    def __init__(self, sequence: str, device: torch.device = None):
        self.sequence = sequence.upper()
        self.num_nucleotides = len(sequence)
        self.device = device or torch.device('cpu')
        self._build()

    def _find_glycosidic_atom(self, mol, base):
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                if base in ['A', 'G']:
                    if [n.GetSymbol() for n in atom.GetNeighbors()].count('C') >= 2:
                        return atom.GetIdx()
                else:
                    if atom.IsInRing():
                        return atom.GetIdx()
        return 0

    def _build(self):
        # 분리된 특징 저장
        physics_features = []      # 불변 물리 특징 (10-dim)
        structural_features = []   # 구조적 특징 (4-dim)
        sugar_features = []
        phosphate_features = []

        bond_edges = [[], []]
        glycosidic_edges, backbone_edges, phos_sugar_edges = [[],[]], [[],[]], [[],[]]

        self.atom_to_nucleotide = []
        self.atom_symbols = []
        self.base_types = []
        self.glycosidic_atoms = []
        self.templates = []
        self.atom_to_template_idx = []
        self.nuc_atom_ranges = []

        atom_offset = 0

        for nuc_idx, base in enumerate(self.sequence):
            mol = Chem.MolFromSmiles(BASE_SMILES[base])
            template = get_initial_template(base)
            self.templates.append(template)

            glyc_local = self._find_glycosidic_atom(mol, base)
            glyc_global = atom_offset + glyc_local
            self.glycosidic_atoms.append(glyc_global)

            start_idx = atom_offset

            for local_idx, atom in enumerate(mol.GetAtoms()):
                phys, struct = get_atom_features(atom)
                physics_features.append(phys)
                structural_features.append(struct)

                self.atom_to_nucleotide.append(nuc_idx)
                self.atom_symbols.append(atom.GetSymbol())
                self.base_types.append(base)
                self.atom_to_template_idx.append(local_idx)

            for bond in mol.GetBonds():
                i, j = atom_offset + bond.GetBeginAtomIdx(), atom_offset + bond.GetEndAtomIdx()
                bond_edges[0].extend([i, j])
                bond_edges[1].extend([j, i])

            atom_offset += mol.GetNumAtoms()
            self.nuc_atom_ranges.append((start_idx, atom_offset))

            sugar_features.append(get_sugar_features())
            phosphate_features.append(get_phosphate_features())

            glycosidic_edges[0].append(nuc_idx)
            glycosidic_edges[1].append(glyc_global)
            phos_sugar_edges[0].append(nuc_idx)
            phos_sugar_edges[1].append(nuc_idx)

            if nuc_idx > 0:
                backbone_edges[0].append(nuc_idx - 1)
                backbone_edges[1].append(nuc_idx)

        self.num_base_atoms = len(physics_features)
        self.num_sugars = len(sugar_features)
        self.num_phosphates = len(phosphate_features)

        # 분리 저장: 물리 특징 (불변) vs 구조 특징
        self.physics_x = torch.tensor(physics_features, dtype=torch.float, device=self.device)
        self.structural_x = torch.tensor(structural_features, dtype=torch.float, device=self.device)
        self.base_atom_x = torch.cat([self.physics_x, self.structural_x], dim=-1)

        self.sugar_x = torch.tensor(sugar_features, dtype=torch.float, device=self.device)
        self.phosphate_x = torch.tensor(phosphate_features, dtype=torch.float, device=self.device)

        def to_edge(e):
            return torch.tensor(e, dtype=torch.long, device=self.device) if e[0] else torch.zeros(2, 0, dtype=torch.long, device=self.device)

        self.bond_edge_index = to_edge(bond_edges)
        self.glycosidic_edge_index = to_edge(glycosidic_edges)
        self.backbone_edge_index = to_edge(backbone_edges)
        self.phos_sugar_edge_index = to_edge(phos_sugar_edges)

        self.atom_to_nucleotide = torch.tensor(self.atom_to_nucleotide, dtype=torch.long, device=self.device)
        self.atom_to_template_idx = torch.tensor(self.atom_to_template_idx, dtype=torch.long, device=self.device)

        # 템플릿도 device로 이동
        self.templates = [t.to(self.device) for t in self.templates]

        # 물리 엔진용 특징 (불변 물리 특징에서 추출)
        self.vdw_radii = self.physics_x[:, 2]   # VdW 반경
        self.is_donor = self.physics_x[:, 6]    # H-bond donor
        self.is_acceptor = self.physics_x[:, 7] # H-bond acceptor

    def to(self, device: torch.device):
        """Device 이동"""
        self.device = device
        self.physics_x = self.physics_x.to(device)
        self.structural_x = self.structural_x.to(device)
        self.base_atom_x = self.base_atom_x.to(device)
        self.sugar_x = self.sugar_x.to(device)
        self.phosphate_x = self.phosphate_x.to(device)
        self.bond_edge_index = self.bond_edge_index.to(device)
        self.glycosidic_edge_index = self.glycosidic_edge_index.to(device)
        self.backbone_edge_index = self.backbone_edge_index.to(device)
        self.phos_sugar_edge_index = self.phos_sugar_edge_index.to(device)
        self.atom_to_nucleotide = self.atom_to_nucleotide.to(device)
        self.atom_to_template_idx = self.atom_to_template_idx.to(device)
        self.templates = [t.to(device) for t in self.templates]
        self.vdw_radii = self.vdw_radii.to(device)
        self.is_donor = self.is_donor.to(device)
        self.is_acceptor = self.is_acceptor.to(device)
        return self

    def to_hetero_data(self):
        data = HeteroData()

        # 물리 특징과 구조 특징 분리 저장
        data['base_atom'].physics_x = self.physics_x        # 불변
        data['base_atom'].structural_x = self.structural_x  # 구조
        data['base_atom'].x = self.base_atom_x              # 전체 (참조용)

        data['sugar'].x = self.sugar_x
        data['phosphate'].x = self.phosphate_x

        if self.bond_edge_index.numel() > 0:
            data['base_atom', 'bond', 'base_atom'].edge_index = self.bond_edge_index
        if self.glycosidic_edge_index.numel() > 0:
            data['sugar', 'glycosidic', 'base_atom'].edge_index = self.glycosidic_edge_index
            data['base_atom', 'rev_glycosidic', 'sugar'].edge_index = self.glycosidic_edge_index.flip(0)
        if self.backbone_edge_index.numel() > 0:
            data['sugar', 'backbone', 'sugar'].edge_index = self.backbone_edge_index
        if self.phos_sugar_edge_index.numel() > 0:
            data['phosphate', 'connects', 'sugar'].edge_index = self.phos_sugar_edge_index

        data.num_nucleotides = self.num_nucleotides
        data.sequence = self.sequence
        return data

    def apply_transforms(self, quaternions, translations):
        coords = torch.zeros(self.num_base_atoms, 3, device=self.device)
        for nuc_idx in range(self.num_nucleotides):
            mask = self.atom_to_nucleotide == nuc_idx
            atom_indices = mask.nonzero(as_tuple=True)[0]
            template_indices = self.atom_to_template_idx[atom_indices]
            local = self.templates[nuc_idx][template_indices]
            coords[atom_indices] = apply_rigid_transform(local, quaternions[nuc_idx], translations[nuc_idx])
        return coords


# ==============================================================================
# 4. SE(3) Rigid Transform
# ==============================================================================

def quaternion_to_rotation_matrix(q):
    q = F.normalize(q, dim=-1)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w], dim=-1),
        torch.stack([2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w], dim=-1),
        torch.stack([2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y], dim=-1),
    ], dim=-2)


def apply_rigid_transform(coords, rotation, translation):
    if rotation.shape[-1] == 4:
        R = quaternion_to_rotation_matrix(rotation)
        if R.dim() == 3:
            R = R.squeeze(0)
    else:
        R = rotation
    return coords @ R.T + translation


# ==============================================================================
# 5. 물리 엔진 (불변 특징 사용)
# ==============================================================================

class DifferentiablePhysicsEngine:
    def __init__(self):
        self.hbond_geo = HBOND_GEOMETRY

    def compute_clash_loss(self, coords, vdw_radii, nuc_ids, eps=1e-8):
        """VdW 충돌 계산 - vdw_radii는 불변 물리 특징"""
        N = coords.size(0)
        dist = torch.cdist(coords, coords) + eps
        min_allowed = (vdw_radii.unsqueeze(0) + vdw_radii.unsqueeze(1)) * 0.75
        same_nuc = nuc_ids.unsqueeze(0) == nuc_ids.unsqueeze(1)
        exclude = same_nuc | torch.eye(N, dtype=torch.bool, device=coords.device)
        pen = F.relu(min_allowed - dist).masked_fill(exclude, 0.0)
        return (pen ** 2).sum() / 2, (pen > 0.1).sum() / 2

    def detect_hbonds(self, coords, is_donor, is_acceptor, nuc_ids, eps=1e-8):
        """H-bond 검출 - is_donor, is_acceptor는 불변 물리 특징"""
        dist = torch.cdist(coords, coords) + eps

        # 물리적으로 가능한 쌍만 (donor ↔ acceptor)
        valid = ((is_donor.unsqueeze(1) * is_acceptor.unsqueeze(0)) +
                 (is_acceptor.unsqueeze(1) * is_donor.unsqueeze(0))).clamp(max=1.0)
        valid = valid * (nuc_ids.unsqueeze(0) != nuc_ids.unsqueeze(1)).float()

        d_opt, (d_min, d_max) = self.hbond_geo['optimal_distance'], self.hbond_geo['distance_range']
        score = torch.exp(-((dist - d_opt) / ((d_max - d_opt) / 2)) ** 2)
        score = score * ((dist >= d_min) & (dist <= d_max)).float()
        strength = score * valid
        return strength, -5.0 * strength.sum() / 2, (strength > 0.5).sum() / 2

    def evaluate(self, coords, vdw_radii, is_donor, is_acceptor, nuc_ids):
        clash_loss, clash_count = self.compute_clash_loss(coords, vdw_radii, nuc_ids)
        hbond_strength, hbond_energy, hbond_count = self.detect_hbonds(coords, is_donor, is_acceptor, nuc_ids)
        return {
            'clash_loss': clash_loss, 'clash_count': clash_count,
            'hbond_strength': hbond_strength, 'hbond_energy': hbond_energy, 'hbond_count': hbond_count,
        }


# ==============================================================================
# 6. 핵심: 불변 특징 + 학습 임베딩 분리
# ==============================================================================

class AtomEmbedding(nn.Module):
    """
    원자 임베딩 - 불변 특징과 학습 특징 분리

    구조:
    - physics_x (10-dim): 불변, 학습 X
    - structural_x (4-dim): 구조적 특징
    - learnable_emb (hidden-dim): 학습 가능한 문맥 임베딩

    출력:
    - full_repr = concat(physics_x, learnable_emb)  # 물리 특징 항상 유지
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 구조적 특징 → 학습 임베딩 초기화
        self.structural_encoder = nn.Sequential(
            nn.Linear(STRUCTURAL_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 물리 특징을 "참조"하여 학습 임베딩에 영향
        self.physics_bias = nn.Linear(INVARIANT_PHYSICS_DIM, hidden_dim, bias=False)

    def forward(self, physics_x: torch.Tensor, structural_x: torch.Tensor):
        """
        Args:
            physics_x: [N, 10] 불변 물리 특징
            structural_x: [N, 4] 구조적 특징

        Returns:
            physics_x: [N, 10] 불변 (그대로 반환)
            learnable_emb: [N, hidden] 학습 가능한 임베딩
        """
        learnable_emb = self.structural_encoder(structural_x)
        learnable_emb = learnable_emb + self.physics_bias(physics_x)
        return physics_x, learnable_emb


class PhysicsAwareGNN(nn.Module):
    """
    물리 특징을 인식하는 GNN

    핵심:
    - Query/Key 계산: 불변 물리 특징 + 학습 임베딩 모두 사용
    - Value 업데이트: 학습 임베딩만 업데이트
    - 출력: 불변 물리 특징은 그대로 유지
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        total_dim = INVARIANT_PHYSICS_DIM + hidden_dim

        self.q_proj = nn.Linear(total_dim, hidden_dim)
        self.k_proj = nn.Linear(total_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        physics_x: torch.Tensor,
        learnable_emb: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return learnable_emb

        full_repr = torch.cat([physics_x, learnable_emb], dim=-1)

        Q = self.q_proj(full_repr)
        K = self.k_proj(full_repr)
        V = self.v_proj(learnable_emb)

        src, dst = edge_index

        attn_scores = (Q[dst] * K[src]).sum(dim=-1) / math.sqrt(Q.size(-1))

        attn_weights = torch.zeros(learnable_emb.size(0), device=learnable_emb.device)
        attn_weights.scatter_add_(0, dst, torch.exp(attn_scores))
        attn_probs = torch.exp(attn_scores) / (attn_weights[dst] + 1e-8)

        msg = V[src] * attn_probs.unsqueeze(-1)
        out = torch.zeros_like(learnable_emb)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        out = self.out_proj(out)
        learnable_emb_updated = self.norm(learnable_emb + out)

        return learnable_emb_updated


def compute_physics_interaction_edges(
    physics_x: torch.Tensor,
    structural_x: torch.Tensor,
    atom_to_nuc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    물리적으로 상호작용 가능한 원자 쌍을 Edge로 반환 (Sparse)
    
    Returns:
        edge_index: [2, E] 유효한 쌍의 (src, dst) 인덱스
        edge_type: [E] 0=hbond, 1=stacking
        total_pairs: N² (비교용)
    """
    N = physics_x.size(0)
    device = physics_x.device
    
    is_donor = physics_x[:, 6]
    is_acceptor = physics_x[:, 7]
    is_aromatic = structural_x[:, 1]
    
    # 모든 쌍 생성 (i, j)
    i_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, N).reshape(-1)
    j_idx = torch.arange(N, device=device).unsqueeze(0).expand(N, N).reshape(-1)
    
    # 조건 계산
    diff_nuc = atom_to_nuc[i_idx] != atom_to_nuc[j_idx]
    
    # H-bond 가능: donor[i] & acceptor[j] or acceptor[i] & donor[j]
    hbond_possible = ((is_donor[i_idx] > 0) & (is_acceptor[j_idx] > 0)) | \
                     ((is_acceptor[i_idx] > 0) & (is_donor[j_idx] > 0))
    
    # Stacking 가능: aromatic[i] & aromatic[j]
    stacking_possible = (is_aromatic[i_idx] > 0) & (is_aromatic[j_idx] > 0)
    
    # 유효한 쌍: (H-bond OR Stacking) AND 다른 뉴클레오타이드
    valid = (hbond_possible | stacking_possible) & diff_nuc
    
    # Edge 추출
    valid_indices = valid.nonzero(as_tuple=True)[0]
    src = i_idx[valid_indices]
    dst = j_idx[valid_indices]
    
    # Edge type: 0=hbond, 1=stacking (hbond 우선)
    edge_type = torch.where(
        hbond_possible[valid_indices],
        torch.zeros_like(src),
        torch.ones_like(src)
    )
    
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index, edge_type, N * N


class SparsePhysicsCrossAttention(nn.Module):
    """
    Sparse Physics-Masked Cross-Attention
    
    핵심 최적화:
    - 물리적으로 가능한 E개 쌍에 대해서만 attention 계산
    - 연산량: O(N²d) → O(Ed) where E << N²
    - 메모리: O(N²H) → O(EH)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        total_dim = INVARIANT_PHYSICS_DIM + hidden_dim

        self.q_proj = nn.Linear(total_dim, hidden_dim)
        self.k_proj = nn.Linear(total_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        physics_x: torch.Tensor,
        structural_x: torch.Tensor,
        learnable_emb: torch.Tensor,
        atom_to_nuc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            learnable_emb_updated: [N, hidden]
            atom_pair_scores: [E] sparse attention scores
            stats: dict with sparsity statistics
        """
        N = physics_x.size(0)
        device = physics_x.device
        
        # 1. 유효한 Edge 추출 (Sparse!)
        edge_index, edge_type, total_pairs = compute_physics_interaction_edges(
            physics_x, structural_x, atom_to_nuc
        )
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        
        # Edge가 없으면 early return
        if E == 0:
            return learnable_emb, torch.zeros(0, device=device), {
                'num_edges': 0, 'total_pairs': total_pairs, 'sparsity': 1.0
            }
        
        # 2. Q, K, V projection (모든 노드에 대해 - O(Nd))
        full_repr = torch.cat([physics_x, learnable_emb], dim=-1)
        
        Q = self.q_proj(full_repr).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(full_repr).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(learnable_emb).view(N, self.num_heads, self.head_dim)
        
        # 3. Edge-based Attention (O(Ed) - Sparse!)
        # Q[dst] dot K[src] for each edge
        q_edge = Q[dst]  # [E, H, d]
        k_edge = K[src]  # [E, H, d]
        v_edge = V[src]  # [E, H, d]
        
        # Attention scores: [E, H]
        attn_scores = (q_edge * k_edge).sum(dim=-1) / self.scale
        
        # 4. Softmax per destination node (scatter softmax)
        # exp(scores) / sum(exp(scores)) for each dst node
        attn_exp = torch.exp(attn_scores - attn_scores.max())  # numerical stability
        
        # Sum of exp per destination
        sum_exp = torch.zeros(N, self.num_heads, device=device)
        sum_exp.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.num_heads), attn_exp)
        
        # Normalize
        attn_probs = attn_exp / (sum_exp[dst] + 1e-8)  # [E, H]
        
        # 5. Weighted value aggregation (O(Ed) - Sparse!)
        weighted_v = attn_probs.unsqueeze(-1) * v_edge  # [E, H, d]
        
        # Scatter add to destination nodes
        out = torch.zeros(N, self.num_heads, self.head_dim, device=device)
        out.scatter_add_(
            0, 
            dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim),
            weighted_v
        )
        
        # 6. Output projection
        out = out.reshape(N, self.hidden_dim)
        out = self.out_proj(out)
        
        # 7. Residual + Norm
        learnable_emb_updated = self.norm(learnable_emb + out)
        
        # 8. Return sparse attention scores
        atom_pair_scores = attn_probs.mean(dim=-1)  # [E]
        
        stats = {
            'num_edges': E,
            'total_pairs': total_pairs,
            'sparsity': 1.0 - E / total_pairs,
            'edge_index': edge_index,
        }
        
        return learnable_emb_updated, atom_pair_scores, stats


# Legacy wrapper for compatibility
def compute_physics_interaction_mask(
    physics_x: torch.Tensor,
    structural_x: torch.Tensor,
    atom_to_nuc: torch.Tensor,
) -> torch.Tensor:
    """Legacy: Dense mask (for backward compatibility)"""
    N = physics_x.size(0)
    device = physics_x.device
    
    is_donor = physics_x[:, 6]
    is_acceptor = physics_x[:, 7]
    is_aromatic = structural_x[:, 1]

    hbond_mask = (
        (is_donor.unsqueeze(1) * is_acceptor.unsqueeze(0)) +
        (is_acceptor.unsqueeze(1) * is_donor.unsqueeze(0))
    ).clamp(max=1.0) > 0

    stacking_mask = (
        is_aromatic.unsqueeze(1) * is_aromatic.unsqueeze(0)
    ) > 0

    diff_nuc_mask = atom_to_nuc.unsqueeze(0) != atom_to_nuc.unsqueeze(1)

    valid_mask = (hbond_mask | stacking_mask) & diff_nuc_mask

    return valid_mask


# Alias for backward compatibility
PhysicsMaskedCrossAttention = SparsePhysicsCrossAttention


# ==============================================================================
# 7. Transform 예측기
# ==============================================================================

class AtomToNucleotideTransform(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.sugar_enc = nn.Linear(8, hidden_dim)
        self.phos_enc = nn.Linear(8, hidden_dim)

        total_dim = INVARIANT_PHYSICS_DIM + hidden_dim
        self.atom_pool_attn = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.nuc_combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.rotation_head[-1].weight)
        self.rotation_head[-1].bias.data = torch.tensor([1., 0., 0., 0.])
        nn.init.zeros_(self.translation_head[-1].weight)
        nn.init.zeros_(self.translation_head[-1].bias)

    def forward(
        self,
        physics_x: torch.Tensor,
        learnable_emb: torch.Tensor,
        atom_to_nuc: torch.Tensor,
        sugar_x: torch.Tensor,
        phos_x: torch.Tensor,
        num_nucleotides: int,
    ):
        h_sugar = self.sugar_enc(sugar_x)
        h_phos = self.phos_enc(phos_x)

        full_repr = torch.cat([physics_x, learnable_emb], dim=-1)

        nuc_from_atoms = []
        for nuc_idx in range(num_nucleotides):
            mask = atom_to_nuc == nuc_idx
            h_atoms = full_repr[mask]
            learnable_atoms = learnable_emb[mask]

            attn = self.atom_pool_attn(h_atoms)
            attn_weights = F.softmax(attn, dim=0)
            pooled = (attn_weights * learnable_atoms).sum(dim=0)
            nuc_from_atoms.append(pooled)

        nuc_from_atoms = torch.stack(nuc_from_atoms)

        nuc_emb = self.nuc_combine(torch.cat([nuc_from_atoms, h_sugar, h_phos], dim=-1))

        quaternions = F.normalize(self.rotation_head(nuc_emb), dim=-1)
        translations = self.translation_head(nuc_emb)

        return quaternions, translations, nuc_emb


# ==============================================================================
# 8. 통합 모델
# ==============================================================================

class PhysicsMaskedRNAModel(nn.Module):
    """물리 마스킹 기반 RNA 구조 예측 모델"""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_cross_attn_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.atom_embedding = AtomEmbedding(hidden_dim)

        self.intra_gnns = nn.ModuleList([
            PhysicsAwareGNN(hidden_dim) for _ in range(num_gnn_layers)
        ])

        self.cross_attns = nn.ModuleList([
            PhysicsMaskedCrossAttention(hidden_dim, num_heads)
            for _ in range(num_cross_attn_layers)
        ])

        self.transform_predictor = AtomToNucleotideTransform(hidden_dim)

        self.physics = DifferentiablePhysicsEngine()

    def forward(self, graph: HeterogeneousRNAGraph):
        data = graph.to_hetero_data()

        physics_x = data['base_atom'].physics_x
        structural_x = data['base_atom'].structural_x

        physics_x, learnable_emb = self.atom_embedding(physics_x, structural_x)

        bond_edge_index = data['base_atom', 'bond', 'base_atom'].edge_index \
            if ('base_atom', 'bond', 'base_atom') in data.edge_types else torch.zeros(2, 0, dtype=torch.long, device=physics_x.device)

        for gnn in self.intra_gnns:
            learnable_emb = gnn(physics_x, learnable_emb, bond_edge_index)

        atom_pair_scores_list = []
        sparse_stats = None

        for cross_attn in self.cross_attns:
            learnable_emb, atom_pair_scores, sparse_stats = cross_attn(
                physics_x, structural_x, learnable_emb, graph.atom_to_nucleotide
            )
            atom_pair_scores_list.append(atom_pair_scores)

        # Sparse attention scores (E개)
        if atom_pair_scores_list and atom_pair_scores_list[0].numel() > 0:
            final_atom_pair_scores = torch.stack(atom_pair_scores_list).mean(dim=0)
        else:
            final_atom_pair_scores = torch.zeros(0, device=physics_x.device)

        quaternions, translations, nuc_emb = self.transform_predictor(
            physics_x, learnable_emb, graph.atom_to_nucleotide,
            graph.sugar_x, graph.phosphate_x, graph.num_nucleotides
        )

        coords = graph.apply_transforms(quaternions, translations)

        physics_eval = self.physics.evaluate(
            coords=coords,
            vdw_radii=graph.vdw_radii,
            is_donor=graph.is_donor,
            is_acceptor=graph.is_acceptor,
            nuc_ids=graph.atom_to_nucleotide,
        )

        return {
            'coords': coords,
            'quaternions': quaternions,
            'translations': translations,
            'physics_x': physics_x,
            'structural_x': structural_x,
            'learnable_emb': learnable_emb,
            'atom_pair_scores': final_atom_pair_scores,
            'sparse_stats': sparse_stats,  # Sparse attention 통계
            'physics': physics_eval,
            'atom_to_nucleotide': graph.atom_to_nucleotide,
            'num_nucleotides': graph.num_nucleotides,
        }

    def compute_loss(self, outputs, target_coords=None, clash_weight=10.0, hbond_weight=1.0):
        losses = {}
        physics = outputs['physics']

        losses['clash'] = physics['clash_loss'] * clash_weight
        losses['hbond'] = -physics['hbond_energy'] * hbond_weight

        # Sparse attention guide loss
        sparse_stats = outputs['sparse_stats']
        atom_pair_scores = outputs['atom_pair_scores']
        
        if sparse_stats and sparse_stats['num_edges'] > 0 and atom_pair_scores.numel() > 0:
            # H-bond 강도가 높은 edge에 대해 attention guide
            edge_index = sparse_stats['edge_index']
            src, dst = edge_index[0], edge_index[1]
            
            # Edge별 H-bond 강도
            hbond_strength = physics['hbond_strength']
            edge_hbond = hbond_strength[src, dst]
            
            # H-bond 가능한 edge만 guide
            hbond_edges = edge_hbond > 0.1
            if hbond_edges.sum() > 0:
                losses['attn_guide'] = -torch.log(atom_pair_scores[hbond_edges] + 1e-8).mean()
            else:
                losses['attn_guide'] = torch.tensor(0.0, device=outputs['coords'].device)
        else:
            losses['attn_guide'] = torch.tensor(0.0, device=outputs['coords'].device)

        if target_coords is not None:
            pred = outputs['coords'] - outputs['coords'].mean(0)
            tgt = target_coords - target_coords.mean(0)
            losses['rmsd'] = torch.sqrt(((pred - tgt) ** 2).sum(-1).mean())

        losses['total'] = sum(v for k, v in losses.items() if k != 'total' and isinstance(v, torch.Tensor))
        return losses


# ==============================================================================
# 9. RNAStructurePredictor (추가)
# ==============================================================================

class RNAStructurePredictor:
    """RNA 구조 예측을 위한 추론 클래스"""

    def __init__(self, model: PhysicsMaskedRNAModel, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def predict(self, sequence: str) -> dict:
        """
        RNA 서열에서 3D 구조 예측

        Args:
            sequence: RNA 서열 (예: "GCAU")

        Returns:
            dict: 예측 결과
        """
        with torch.no_grad():
            graph = HeterogeneousRNAGraph(sequence, device=self.device)
            outputs = self.model(graph)

            # Sparse stats
            sparse_stats = outputs['sparse_stats']
            sparsity_info = {
                'num_edges': sparse_stats['num_edges'] if sparse_stats else 0,
                'total_pairs': sparse_stats['total_pairs'] if sparse_stats else 0,
                'sparsity': sparse_stats['sparsity'] if sparse_stats else 1.0,
            }

            return {
                'sequence': sequence,
                'coords': outputs['coords'].cpu().numpy(),
                'quaternions': outputs['quaternions'].cpu().numpy(),
                'translations': outputs['translations'].cpu().numpy(),
                'atom_pair_scores': outputs['atom_pair_scores'].cpu().numpy(),
                'physics': {
                    'clash_loss': outputs['physics']['clash_loss'].item(),
                    'clash_count': outputs['physics']['clash_count'].item(),
                    'hbond_energy': outputs['physics']['hbond_energy'].item(),
                    'hbond_count': outputs['physics']['hbond_count'].item(),
                },
                'sparsity': sparsity_info,
                'atom_info': {
                    'symbols': graph.atom_symbols,
                    'base_types': graph.base_types,
                    'nucleotide_ids': graph.atom_to_nucleotide.cpu().numpy(),
                },
                'num_nucleotides': graph.num_nucleotides,
            }


# ==============================================================================
# 10. 데이터셋
# ==============================================================================

class RNADataset:
    """CSV 데이터셋 로드 및 전처리"""

    def __init__(self, sequences_path: str, labels_path: str, max_samples: int = None):
        self.sequences_df = pd.read_csv(sequences_path)
        self.labels_df = pd.read_csv(labels_path)

        self.target_ids = self.sequences_df['target_id'].unique().tolist()

        if max_samples:
            self.target_ids = self.target_ids[:max_samples]

        print(f"Loaded {len(self.target_ids)} RNA samples")

        self.data_cache = {}
        self._preprocess()

    def _preprocess(self):
        for target_id in self.target_ids:
            seq_row = self.sequences_df[self.sequences_df['target_id'] == target_id]
            if len(seq_row) == 0:
                continue

            sequence = seq_row['sequence'].values[0]

            label_rows = self.labels_df[self.labels_df['ID'].str.startswith(f'{target_id}_')]
            label_rows = label_rows.sort_values('resid')

            if len(label_rows) == 0:
                continue

            coords = []
            for _, row in label_rows.iterrows():
                atom_coords = []
                for i in range(1, 41):
                    x = row.get(f'x_{i}', -1e18)
                    y = row.get(f'y_{i}', -1e18)
                    z = row.get(f'z_{i}', -1e18)

                    if x > -1e17 and y > -1e17 and z > -1e17:
                        atom_coords.append([x, y, z])

                if atom_coords:
                    centroid = np.mean(atom_coords, axis=0)
                    coords.append(centroid)
                else:
                    coords.append([0.0, 0.0, 0.0])

            coords = np.array(coords, dtype=np.float32)

            min_len = min(len(sequence), len(coords))
            sequence = sequence[:min_len]
            coords = coords[:min_len]

            self.data_cache[target_id] = {
                'sequence': sequence,
                'coords': coords,
            }

        self.target_ids = [tid for tid in self.target_ids if tid in self.data_cache]
        print(f"Valid samples after preprocessing: {len(self.target_ids)}")

    def __len__(self):
        return len(self.target_ids)

    def __getitem__(self, idx):
        target_id = self.target_ids[idx]
        data = self.data_cache[target_id]

        return {
            'target_id': target_id,
            'sequence': data['sequence'],
            'coords': torch.tensor(data['coords'], dtype=torch.float32),
        }


# ==============================================================================
# 11. 학습 함수
# ==============================================================================

def compute_rmsd_loss(pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """RMSD 계산"""
    pred_center = pred_coords.mean(dim=0, keepdim=True)
    target_center = target_coords.mean(dim=0, keepdim=True)

    pred_centered = pred_coords - pred_center
    target_centered = target_coords - target_center

    diff = pred_centered - target_centered
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())

    return rmsd


def compute_supervised_loss(
    outputs: dict,
    target_nuc_coords: torch.Tensor,
    rmsd_weight: float = 1.0,
    clash_weight: float = 0.1,
    hbond_weight: float = 0.05,
) -> dict:
    """지도 학습 손실 함수"""
    pred_coords = outputs['coords']
    atom_to_nuc = outputs['atom_to_nucleotide']

    num_nuc = outputs['num_nucleotides']
    pred_nuc_coords = []
    for nuc_id in range(num_nuc):
        mask = atom_to_nuc == nuc_id
        if mask.sum() > 0:
            nuc_center = pred_coords[mask].mean(dim=0)
            pred_nuc_coords.append(nuc_center)

    pred_nuc_coords = torch.stack(pred_nuc_coords)

    min_len = min(len(pred_nuc_coords), len(target_nuc_coords))
    pred_nuc_coords = pred_nuc_coords[:min_len]
    target_nuc_coords = target_nuc_coords[:min_len]

    rmsd_loss = compute_rmsd_loss(pred_nuc_coords, target_nuc_coords)

    physics = outputs['physics']
    clash_loss = physics['clash_loss']
    hbond_loss = -physics['hbond_energy'] * 0.01

    total_loss = (
        rmsd_weight * rmsd_loss +
        clash_weight * clash_loss +
        hbond_weight * hbond_loss
    )

    return {
        'total': total_loss,
        'rmsd': rmsd_loss,
        'clash': clash_loss,
        'hbond': hbond_loss,
    }


def train_with_real_data(
    model: PhysicsMaskedRNAModel,
    dataset: RNADataset,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cpu',
):
    """실제 RNA 데이터로 학습"""
    device = torch.device(device)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'epoch': [],
        'total_loss': [],
        'rmsd': [],
        'clash': [],
        'hbond': [],
    }

    print("=" * 70)
    print("  RNA 3D Structure Training")
    print("=" * 70)
    print(f"  Samples: {len(dataset)}, Epochs: {num_epochs}, LR: {lr}")
    print("=" * 70)

    for epoch in range(num_epochs):
        epoch_losses = {'total': 0, 'rmsd': 0, 'clash': 0, 'hbond': 0}
        valid_samples = 0

        for idx in range(len(dataset)):
            sample = dataset[idx]
            sequence = sample['sequence']
            target_coords = sample['coords'].to(device)

            try:
                # 그래프 생성 (device 지정)
                graph = HeterogeneousRNAGraph(sequence, device=device)

                # Forward
                outputs = model(graph)

                # Loss 계산
                losses = compute_supervised_loss(outputs, target_coords)

                # Backward
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 기록
                epoch_losses['total'] += losses['total'].item()
                epoch_losses['rmsd'] += losses['rmsd'].item()
                epoch_losses['clash'] += losses['clash'].item()
                epoch_losses['hbond'] += losses['hbond'].item()
                valid_samples += 1

            except Exception as e:
                print(f"  Skip {sample['target_id']}: {e}")
                continue

        if valid_samples > 0:
            for k in epoch_losses:
                epoch_losses[k] /= valid_samples

            history['epoch'].append(epoch + 1)
            history['total_loss'].append(epoch_losses['total'])
            history['rmsd'].append(epoch_losses['rmsd'])
            history['clash'].append(epoch_losses['clash'])
            history['hbond'].append(epoch_losses['hbond'])

            print(f"Epoch {epoch+1:3d} | Loss: {epoch_losses['total']:.4f} | "
                  f"RMSD: {epoch_losses['rmsd']:.2f}Å | "
                  f"Clash: {epoch_losses['clash']:.2f} | "
                  f"H-bond: {epoch_losses['hbond']:.4f}")

    print("=" * 70)
    print("Training Complete!")

    return history


# ==============================================================================
# 12. 평가 함수
# ==============================================================================

def evaluate_all_samples(model: PhysicsMaskedRNAModel, dataset: RNADataset, device: str = 'cpu'):
    """모든 샘플에 대해 RMSD 평가"""
    device = torch.device(device)
    model.eval()
    predictor = RNAStructurePredictor(model, device=device)

    results = []

    print("\n[전체 샘플 평가]")
    print("-" * 60)

    for idx in range(len(dataset)):
        sample = dataset[idx]

        try:
            result = predictor.predict(sample['sequence'])
            target_coords = sample['coords'].numpy()

            pred_coords = result['coords']
            nuc_ids = result['atom_info']['nucleotide_ids']

            pred_nuc_coords = []
            for nuc_id in range(result['num_nucleotides']):
                mask = nuc_ids == nuc_id
                if mask.sum() > 0:
                    pred_nuc_coords.append(pred_coords[mask].mean(axis=0))
            pred_nuc_coords = np.array(pred_nuc_coords)

            min_len = min(len(pred_nuc_coords), len(target_coords))
            diff = pred_nuc_coords[:min_len] - target_coords[:min_len]
            rmsd = np.sqrt((diff ** 2).sum(axis=-1).mean())

            results.append({
                'id': sample['target_id'],
                'rmsd': rmsd,
                'hbond': result['physics']['hbond_count'],
                'clash': result['physics']['clash_count'],
            })

            print(f"{sample['target_id']:<15} RMSD: {rmsd:>8.2f} Å  "
                  f"H-bonds: {result['physics']['hbond_count']:>4.0f}  "
                  f"Clashes: {result['physics']['clash_count']:>4.0f}")

        except Exception as e:
            print(f"{sample['target_id']:<15} Error: {e}")

    print("-" * 60)

    if results:
        avg_rmsd = np.mean([r['rmsd'] for r in results])
        print(f"Average RMSD: {avg_rmsd:.2f} Å")

    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 테스트
    # print("\n" + "=" * 70)
    # print("  Sparse Physics Attention Test")
    # print("=" * 70)

    # sequence = "GCAU"
    # graph = HeterogeneousRNAGraph(sequence, device=device)
    # model = PhysicsMaskedRNAModel(hidden_dim=64).to(device)

    # print(f"\n[그래프]")
    # print(f"  서열: {sequence}")
    # print(f"  원자 수 (N): {graph.num_base_atoms}")
    # print(f"  전체 쌍 (N²): {graph.num_base_atoms ** 2}")

    # model.eval()
    # with torch.no_grad():
    #     outputs = model(graph)

    # # Sparse Attention 통계
    # sparse_stats = outputs['sparse_stats']
    
    # print(f"\n[Sparse Attention 효과]")
    # print(f"  전체 원자 쌍 (Dense): {sparse_stats['total_pairs']:,}")
    # print(f"  유효한 Edge (Sparse): {sparse_stats['num_edges']:,}")
    # print(f"  Sparsity: {sparse_stats['sparsity']*100:.1f}%")
    # print(f"  연산량 절감: {sparse_stats['sparsity']*100:.1f}%")

    # # 연산량 비교
    # N = graph.num_base_atoms
    # E = sparse_stats['num_edges']
    # d = 64  # hidden_dim
    
    # dense_flops = N * N * d  # O(N²d)
    # sparse_flops = E * d      # O(Ed)
    
    # print(f"\n[연산량 비교 (Cross-Attention)]")
    # print(f"  Dense FLOPs:  {dense_flops:,}")
    # print(f"  Sparse FLOPs: {sparse_flops:,}")
    # print(f"  절감율: {(1 - sparse_flops/dense_flops)*100:.1f}%")

    # # 메모리 비교
    # dense_mem = N * N * 4 * 4  # N² × num_heads × float32
    # sparse_mem = E * 4 * 4      # E × num_heads × float32
    
    # print(f"\n[메모리 비교 (Attention Scores)]")
    # print(f"  Dense Memory:  {dense_mem/1024:.1f} KB")
    # print(f"  Sparse Memory: {sparse_mem/1024:.1f} KB")
    # print(f"  절감율: {(1 - sparse_mem/dense_mem)*100:.1f}%")

    # # 불변성 검증
    # print(f"\n[불변성 검증]")
    # physics_before = graph.physics_x.clone()
    # physics_after = outputs['physics_x']
    # is_invariant = torch.equal(physics_before, physics_after)
    # print(f"  물리 특징 불변: {is_invariant} ✅" if is_invariant else "  물리 특징 변경됨 ❌")

    # # 더 긴 서열 테스트
    # print("\n" + "=" * 70)
    # print("  서열 길이별 Sparsity 테스트")
    # print("=" * 70)
    
    # test_sequences = ["GCAU", "GCAUGCAU", "GCAUGCAUGCAUGCAU", "GCAU" * 10]
    
    # for seq in test_sequences:
    #     g = HeterogeneousRNAGraph(seq, device=device)
    #     with torch.no_grad():
    #         out = model(g)
    #     stats = out['sparse_stats']
    #     print(f"  L={len(seq):3d}, N={g.num_base_atoms:4d}, "
    #           f"N²={g.num_base_atoms**2:6d}, E={stats['num_edges']:5d}, "
    #           f"Sparsity={stats['sparsity']*100:5.1f}%")

    # print("\n" + "=" * 70)
    # print("  테스트 완료!")
    # print("=" * 70)
    SEQUENCES_PATH = 'sequences.csv'
    LABELS_PATH = 'labels.csv'

    dataset = RNADataset(SEQUENCES_PATH, LABELS_PATH, max_samples=10)
    model = PhysicsMaskedRNAModel(hidden_dim=64)

    # 학습 실행
    history = train_with_real_data(
        model=model,
        dataset=dataset,
        num_epochs=20,
        lr=1e-3,
)

