# RNA Foundation Model v9 - Physics-Masked Cross-Attention

이 문서는 RNA 2D/3D 구조 예측을 위한 **Physics-Masked Cross-Attention** 기반 Foundation Model의 설계와 구현을 설명합니다.

---

## 1. 핵심 철학

### 1.1 세 가지 핵심 분리

| 구분 | 내용 | 학습 여부 |
|------|------|----------|
| **불변 물리 특징** | 원자번호, VdW 반경, H-bond donor/acceptor | ❌ 고정 |
| **학습 임베딩** | 문맥 정보, 상호작용 패턴 | ✅ 학습 |
| **물리 엔진** | 충돌 감지, H-bond 판별 | ❌ 미분 가능 |

### 1.2 불변 특징 vs 학습 임베딩

**핵심 원칙:** 물리 상수는 절대 학습되지 않음

```
원자 표현 = [불변 물리 특징] + [학습 임베딩]
              (freeze)           (update)

┌──────────────────┬──────────────────────────────────────┐
│ 불변 물리 (10d)  │ 학습 임베딩 (hidden)                 │
├──────────────────┼──────────────────────────────────────┤
│ 원자번호         │ 문맥 정보                            │
│ VdW 반경         │ 상호작용 패턴                        │
│ 전기음성도       │ 구조적 역할                          │
│ donor/acceptor   │                                      │
├──────────────────┼──────────────────────────────────────┤
│ 용도:            │ 용도:                                │
│ - 물리 마스킹    │ - GNN 업데이트                       │
│ - Q/K 계산 참조  │ - Cross-Attention                    │
│ - 물리 엔진      │ - R, T 예측                          │
├──────────────────┼──────────────────────────────────────┤
│ 학습 ❌          │ 학습 ✅                              │
└──────────────────┴──────────────────────────────────────┘
```

### 1.3 물리 마스킹 (Physics Masking)

**핵심 최적화:** 물리적으로 불가능한 원자 쌍은 Attention에서 제외

```
┌─────────────────────────────────────────────────────────────┐
│                    Attention 대상 (마스킹 X)                │
├─────────────────────────────────────────────────────────────┤
│  H-bond 가능: donor ↔ acceptor                             │
│  Stacking 가능: aromatic ↔ aromatic                        │
├─────────────────────────────────────────────────────────────┤
│                    마스킹됨 (Attention = 0)                 │
├─────────────────────────────────────────────────────────────┤
│  C ↔ C (H-bond 불가, stacking 불가)                        │
│  같은 뉴클레오타이드 내 원자                                │
└─────────────────────────────────────────────────────────────┘
```

**효과:**
- 계산 효율: O(N²) → O(가능한 쌍만)
- 물리적 의미: 실제 상호작용 가능한 쌍에만 집중
- 노이즈 감소: 불필요한 attention이 학습을 방해하지 않음

### 1.4 Physics Loss의 역할

**마스킹과 Physics Loss는 다른 역할:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        물리 마스킹                              │
├─────────────────────────────────────────────────────────────────┤
│ 역할: "학습 시 어디에 집중할지" (attention 방향)                 │
│ 적용: Attention 계산 전                                        │
│ 대상: H-bond/Stacking 가능 쌍                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     R, T 예측 → 좌표 생성
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Physics Loss                               │
├─────────────────────────────────────────────────────────────────┤
│ 역할: "예측된 좌표가 물리적으로 타당한지" (결과 검증)            │
│ 적용: 좌표 예측 후                                             │
│ 대상: 모든 원자 쌍 (C-C 충돌도 감지)                           │
└─────────────────────────────────────────────────────────────────┘
```

**왜 둘 다 필요한가?**
- **Clash**: 마스킹과 무관하게 모든 원자 쌍에서 발생 가능
- **H-bond**: 가능 쌍이어도 거리가 멀면 안 됨 → 거리 검증 필요

---

## 2. 데이터 표현

### 2.1 원자 특징 분류

**불변 물리 특징 (10차원) - 절대 학습 X:**

```
[0]  원자 번호 (H=1, C=6, N=7, O=8) - 자연 상수
[1]  질량 (g/mol) - 자연 상수
[2]  VdW 반경 (Å) - 자연 상수, 충돌 판정용
[3]  VdW 부피 (Å³) - 자연 상수
[4]  전기음성도 - 자연 상수
[5]  전기음성 원자 여부 (N, O) - 화학적 정의
[6]  H-bond Donor (N-H, O-H 보유) - 화학적 정의
[7]  H-bond Acceptor (N, O lone pair) - 화학적 정의
[8]  부착된 H 개수
[9]  형식 전하
```

**구조적 특징 (4차원) - 학습 임베딩 초기화용:**

```
[0]  혼성화 (SP=1, SP2=2, SP3=3)
[1]  방향족성 (π-전자 시스템) - Stacking 마스킹용
[2]  고리 내 여부
[3]  결합 차수
```

**당 노드 (8차원):**

```
[0]  노드 타입: 1.0 (sugar)
[1]  Sugar Pucker C2'-endo
[2]  Sugar Pucker C3'-endo
[3]  D-ribose 치랄리티
[4]  분자량
[5]  VdW 부피
[6]  탄소 개수
[7]  산소 개수
```

**인산 노드 (8차원):**

```
[0]  노드 타입: 2.0 (phosphate)
[1]  순 전하
[2]  분자량
[3]  VdW 부피
[4]  P 개수
[5]  O 개수
[6]  P VdW 반경
[7]  P 전기음성도
```

### 2.2 그래프 구조

```
RNA Sequence (예: GCAU)
       ┌──────────────────────────────────────────────────┐
       │  이질적 그래프 (HeteroData)                       │
       │                                                  │
       │  노드 타입:                                      │
       │    - base_atom: 염기 원자 (14차원 = 10+4)        │
       │    - sugar: 당 (8차원)                          │
       │    - phosphate: 인산 (8차원)                    │
       │                                                  │
       │  에지 타입:                                      │
       │    - bond: 염기 내 화학 결합                     │
       │    - glycosidic: 당 → N1/N9 연결               │
       │    - backbone: 당[i] → 당[i+1]                 │
       │    - connects: 인산 → 당                       │
       └──────────────────────────────────────────────────┘
```

---

## 3. 모델 아키텍처

### 3.1 전체 파이프라인

```
입력: RNA 서열
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. HeterogeneousRNAGraph                                    │
│    서열 → 이질적 그래프 변환                                 │
│    physics_x (10d) + structural_x (4d) 분리 저장            │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. AtomEmbedding                                            │
│    physics_x → 그대로 유지 (학습 X)                         │
│    structural_x → learnable_emb 초기화 (학습 O)             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PhysicsAwareGNN (× N layers)                             │
│    서브그래프 내부 메시지 패싱 (염기 내 화학 결합)           │
│    Q/K: physics_x + learnable_emb 사용                      │
│    V: learnable_emb만 업데이트                              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PhysicsMaskedCrossAttention (× M layers) ← 핵심!         │
│    서브그래프 간 원자 레벨 Cross-Attention                   │
│    물리 마스킹: H-bond/Stacking 가능 쌍만 attention          │
│    출력: atom_pair_scores, physics_mask                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. AtomToNucleotideTransform                                │
│    원자 임베딩 → 뉴클레오타이드 R, T 예측                    │
│    Attention Pooling (physics_x 참조, learnable_emb 집계)   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 전역 좌표 계산                                           │
│    템플릿에 R, T 적용 → 전역 좌표                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. DifferentiablePhysicsEngine                              │
│    충돌 감지: 모든 원자 쌍 (불변 VdW 반경 사용)              │
│    H-bond 판별: donor/acceptor 쌍 + 거리 조건               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
출력: coords, physics_eval, atom_pair_scores, physics_mask
```

### 3.2 AtomEmbedding (불변/학습 분리)

**목적:** 물리 특징은 유지하고, 학습 임베딩만 생성

```python
class AtomEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        # 구조적 특징 → 학습 임베딩
        self.structural_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # 물리 특징은 bias로만 영향 (값 자체는 변환 X)
        self.physics_bias = nn.Linear(10, hidden_dim, bias=False)
    
    def forward(self, physics_x, structural_x):
        # physics_x: [N, 10] 불변
        # structural_x: [N, 4] 구조
        
        learnable_emb = self.structural_encoder(structural_x)
        learnable_emb = learnable_emb + self.physics_bias(physics_x)
        
        # physics_x는 그대로 반환 (절대 변경 X)
        return physics_x, learnable_emb
```

### 3.3 PhysicsAwareGNN (물리 인식 GNN)

**목적:** 서브그래프 내부 메시지 패싱, 물리 특징 유지

```python
class PhysicsAwareGNN(nn.Module):
    def forward(self, physics_x, learnable_emb, edge_index):
        # 전체 표현 (Q, K 계산용)
        full_repr = concat(physics_x, learnable_emb)  # [N, 10+hidden]
        
        Q = self.q_proj(full_repr)  # 물리 특징 참조
        K = self.k_proj(full_repr)  # 물리 특징 참조
        V = self.v_proj(learnable_emb)  # 학습 임베딩만!
        
        # Edge-based attention
        # ...
        
        # 학습 임베딩만 업데이트
        learnable_emb_updated = self.norm(learnable_emb + out)
        
        return learnable_emb_updated  # physics_x는 호출자가 유지
```

### 3.4 PhysicsMaskedCrossAttention (핵심!)

**목적:** 서브그래프 간 원자 레벨 Cross-Attention, 물리적으로 가능한 쌍만

```python
def compute_physics_interaction_mask(physics_x, structural_x, atom_to_nuc):
    """물리적으로 상호작용 가능한 원자 쌍 마스크"""
    
    is_donor = physics_x[:, 6]      # H-bond donor
    is_acceptor = physics_x[:, 7]   # H-bond acceptor
    is_aromatic = structural_x[:, 1]  # 방향족성
    
    # H-bond 가능: donor ↔ acceptor
    hbond_mask = (donor_i * acceptor_j) | (acceptor_i * donor_j)
    
    # Stacking 가능: aromatic ↔ aromatic
    stacking_mask = aromatic_i * aromatic_j
    
    # 다른 뉴클레오타이드 간만
    diff_nuc_mask = atom_to_nuc_i != atom_to_nuc_j
    
    # 최종: (H-bond OR Stacking) AND 다른 뉴클레오타이드
    return (hbond_mask | stacking_mask) & diff_nuc_mask


class PhysicsMaskedCrossAttention(nn.Module):
    def forward(self, physics_x, structural_x, learnable_emb, atom_to_nuc):
        # 1. 물리 마스크 계산 (사전에!)
        physics_mask = compute_physics_interaction_mask(
            physics_x, structural_x, atom_to_nuc
        )
        
        # 2. Q, K, V
        full_repr = concat(physics_x, learnable_emb)
        Q = self.q_proj(full_repr)
        K = self.k_proj(full_repr)
        V = self.v_proj(learnable_emb)  # 학습 임베딩만!
        
        # 3. Attention (물리적으로 가능한 쌍만!)
        attn_scores = Q @ K.T / sqrt(d)
        
        # 불가능한 쌍은 -inf
        attn_scores = attn_scores.masked_fill(~physics_mask, -inf)
        
        attn_probs = softmax(attn_scores)
        
        # 4. Value aggregation (학습 임베딩만 업데이트)
        out = attn_probs @ V
        learnable_emb_updated = self.norm(learnable_emb + out)
        
        return learnable_emb_updated, atom_pair_scores, physics_mask
```

### 3.5 AtomToNucleotideTransform

**목적:** 원자 레벨 임베딩 → 뉴클레오타이드 R, T 예측

```python
class AtomToNucleotideTransform(nn.Module):
    def forward(self, physics_x, learnable_emb, atom_to_nuc, sugar_x, phos_x):
        # 전체 표현 (Attention weight 계산용)
        full_repr = concat(physics_x, learnable_emb)
        
        # 뉴클레오타이드별 Attention Pooling
        for nuc_idx in range(num_nucleotides):
            atoms = (atom_to_nuc == nuc_idx)
            
            # Attention weights: physics_x 참조
            attn = self.attn_pool(full_repr[atoms])
            
            # Pooling: learnable_emb만
            nuc_emb[nuc_idx] = (softmax(attn) * learnable_emb[atoms]).sum()
        
        # 당 + 인산 결합
        nuc_emb = combine(nuc_emb, sugar_emb, phos_emb)
        
        # R, T 예측
        quaternions = normalize(self.rotation_head(nuc_emb))
        translations = self.translation_head(nuc_emb)
        
        return quaternions, translations
```

### 3.6 SE(3) Rigid Body Transform

**좌표를 직접 예측하지 않고, R + T만 예측:**

```python
def quaternion_to_rotation_matrix(q):
    """Quaternion [w, x, y, z] → 3x3 Rotation Matrix"""
    q = normalize(q)
    w, x, y, z = q
    
    R = [
        [1 - 2*y² - 2*z², 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x² - 2*z², 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x² - 2*y²],
    ]
    return R


def apply_transforms(templates, quaternions, translations):
    """각 뉴클레오타이드의 로컬 템플릿에 Rigid Transform 적용"""
    for nuc_idx in range(num_nucleotides):
        local = templates[nuc_idx]
        R = quaternion_to_rotation_matrix(quaternions[nuc_idx])
        T = translations[nuc_idx]
        
        global_coords[nuc_idx] = local @ R.T + T
    
    return global_coords
```

### 3.7 DifferentiablePhysicsEngine

**목적:** 예측된 좌표의 물리적 타당성 검증, 불변 물리 특징 사용

```python
class DifferentiablePhysicsEngine:
    def compute_clash_loss(self, coords, vdw_radii, nuc_ids):
        """VdW 충돌 감지 - vdw_radii는 불변 물리 특징"""
        dist = cdist(coords, coords)
        min_allowed = (vdw_radii_i + vdw_radii_j) * 0.75
        
        # 다른 뉴클레오타이드 간만
        diff_nuc = nuc_ids_i != nuc_ids_j
        
        penetration = relu(min_allowed - dist) * diff_nuc
        return (penetration ** 2).sum()
    
    def detect_hbonds(self, coords, is_donor, is_acceptor, nuc_ids):
        """H-bond 판별 - is_donor, is_acceptor는 불변 물리 특징"""
        dist = cdist(coords, coords)
        
        # 물리적으로 가능한 쌍만
        valid = (donor_i * acceptor_j) | (acceptor_i * donor_j)
        valid = valid * (nuc_ids_i != nuc_ids_j)
        
        # 거리 조건 (2.4 ~ 3.5Å)
        in_range = (dist >= 2.4) & (dist <= 3.5)
        
        # 강도 (최적 거리 2.8Å에서 최대)
        strength = exp(-((dist - 2.8) / σ)²) * valid * in_range
        
        return strength, -5.0 * strength.sum()  # 음수 = 안정화
```

---

## 4. Loss 함수

```python
def compute_loss(outputs, target_coords=None):
    losses = {}
    physics = outputs['physics']
    
    # 1. 충돌 Loss (높을수록 나쁨)
    losses['clash'] = physics['clash_loss'] * 10.0
    
    # 2. H-bond Loss (음수 에너지 → 양수 loss로 반전)
    losses['hbond'] = -physics['hbond_energy'] * 1.0
    
    # 3. Attention Guide (물리 마스크 내 H-bond 쌍에 높은 attention 유도)
    physics_mask = outputs['physics_mask']
    hbond_possible = (physics['hbond_strength'] > 0.1) * physics_mask
    if hbond_possible.sum() > 0:
        losses['attn_guide'] = -log(atom_pair_scores + ε) * hbond_possible
        losses['attn_guide'] = losses['attn_guide'].sum() / hbond_possible.sum()
    
    # 4. RMSD (타겟 좌표 있을 때)
    if target_coords is not None:
        pred = outputs['coords'] - outputs['coords'].mean(0)
        tgt = target_coords - target_coords.mean(0)
        losses['rmsd'] = sqrt(((pred - tgt) ** 2).mean())
    
    losses['total'] = sum(losses.values())
    return losses
```

---

## 5. 정보 흐름 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                         입력 처리                               │
├─────────────────────────────────────────────────────────────────┤
│ RNA 서열 → HeterogeneousRNAGraph                                │
│   ├─ physics_x [N, 10]: 불변 물리 특징 (VdW, donor/acceptor)   │
│   └─ structural_x [N, 4]: 구조 특징 (aromatic 등)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AtomEmbedding                             │
├─────────────────────────────────────────────────────────────────┤
│ physics_x ────────────────────────────────────► physics_x (유지)│
│ structural_x ──► encoder ──► learnable_emb ──► learnable_emb    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PhysicsAwareGNN (× N)                        │
├─────────────────────────────────────────────────────────────────┤
│ physics_x + learnable_emb ──► Q, K                              │
│ learnable_emb ──► V                                             │
│ 출력: learnable_emb 업데이트 (physics_x 유지)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PhysicsMaskedCrossAttention (× M)                  │
├─────────────────────────────────────────────────────────────────┤
│ 마스크 계산: physics_x, structural_x ──► physics_mask           │
│   └─ H-bond 가능: donor ↔ acceptor                             │
│   └─ Stacking 가능: aromatic ↔ aromatic                        │
│                                                                 │
│ Attention: 마스킹된 쌍만 계산                                   │
│ 출력: learnable_emb 업데이트, atom_pair_scores                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AtomToNucleotideTransform                       │
├─────────────────────────────────────────────────────────────────┤
│ Attention Pooling: physics_x + learnable_emb ──► weights        │
│ 집계: learnable_emb만                                           │
│ 출력: quaternions [L, 4], translations [L, 3]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      전역 좌표 계산                              │
├─────────────────────────────────────────────────────────────────┤
│ 템플릿 + R, T ──► coords [num_atoms, 3]                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DifferentiablePhysicsEngine                     │
├─────────────────────────────────────────────────────────────────┤
│ Clash: coords + vdw_radii (불변) ──► clash_loss                 │
│ H-bond: coords + donor/acceptor (불변) ──► hbond_strength       │
│                                                                 │
│ 출력: physics_eval (모든 원자 쌍 검사)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 핵심 설계 원칙 정리

### 원칙 1: 불변 물리 특징 분리
- 원자번호, VdW 반경, donor/acceptor는 **절대 학습되지 않음**
- 물리 엔진과 마스킹에서 **원본 값 그대로 사용**

### 원칙 2: 물리 마스킹
- 물리적으로 **불가능한 쌍은 attention에서 제외**
- H-bond: donor ↔ acceptor 쌍만
- Stacking: aromatic ↔ aromatic 쌍만

### 원칙 3: 원자 레벨 Cross-Attention
- 서브그래프(뉴클레오타이드)를 압축하지 않음
- **원자 레벨에서 직접 attention** → 정보 손실 없음
- "i의 N3 원자가 j의 O2 원자와 상호작용" 학습 가능

### 원칙 4: Q/K와 V의 분리
- **Q, K**: physics_x + learnable_emb (물리적 호환성 판단)
- **V**: learnable_emb만 (업데이트 대상)
- 물리 특징은 참조만, 변환되지 않음

### 원칙 5: Physics Loss는 결과 검증
- 마스킹: "어디에 집중할지" (학습 방향)
- Physics Loss: "결과가 물리적으로 타당한지" (검증)
- 둘은 다른 역할, 모두 필요

---

## 7. 구현 파일 구조

```
RNA/
├── RNA_Foundation_Model_Colab.ipynb  # 통합 모델 (Colab용)
├── ARCHITECTURE.md                    # 이 문서
├── models/
│   ├── gnn_encoder.py                # PhysicsAwareGNN
│   ├── foundation_model.py           # PhysicsMaskedRNAModel
│   └── atom_interaction.py           # PhysicsMaskedCrossAttention
├── rna_base_graph.py                 # HeterogeneousRNAGraph
└── test_pipeline.py                  # 테스트
```

---

## 8. 버전 히스토리

| 버전 | 주요 변경 |
|------|----------|
| v1-v4 | 초기 설계, 이질적 그래프 도입 |
| v5 | Pure Physics-Based (Hard Constraints) |
| v6 | Energy Minimization |
| v7 | SE(3) Equivariant (R, T 예측) |
| v8 | Subgraph Interaction Attention |
| **v9** | **Physics-Masked Cross-Attention, 불변 특징 분리** |
