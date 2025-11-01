# HSI Model Design Architecture

## Overview
This document provides a detailed breakdown of the neural network model architecture used in the HSI anomaly detection system, focusing on the **PatchAutoencoder** and **SimpleTransformer** model components.

---

## Model Architecture Components

### 1. PatchAutoencoder Model

The **PatchAutoencoder** is a deep neural network consisting of two main components: an **Encoder** and a **Decoder**, working together to learn compact representations of hyperspectral image patches.

#### Model Structure

```python
class PatchAutoencoder(nn.Module):
    - Encoder: Compresses input patches → latent space (32-dim)
    - Decoder: Reconstructs latent → original patch dimensions
    - Input: Flattened patch (patch_size² × PCA_components)
    - Output: Reconstructed patch + Latent features (Z)
```

---

## Detailed Architecture Flowchart

```mermaid
graph TB
    subgraph "INPUT PREPARATION"
        Input[HSI Patch Input<br/>Shape: batch × input_dim<br/>input_dim = patch_size² × PCA_dim<br/>Example: batch × 4096 if 16×16×16]
    end
    
    subgraph "ENCODER NETWORK"
        Input --> EncLayer1[Linear Layer 1<br/>Weight Matrix: W₁<br/>Shape: input_dim × 512<br/>Bias: b₁ ∈ R^512<br/>Operation: y = xW₁ᵀ + b₁]
        
        EncLayer1 --> EncReLU1[ReLU Activation<br/>f(x) = max(0, x)<br/>Element-wise non-linearity<br/>Shape: batch × 512]
        
        EncReLU1 --> EncLayer2[Linear Layer 2<br/>Weight Matrix: W₂<br/>Shape: 512 × latent_dim<br/>Bias: b₂ ∈ R^32<br/>latent_dim = 32<br/>Operation: z = yW₂ᵀ + b₂]
        
        EncLayer2 --> LatentZ[Latent Representation Z<br/>Shape: batch × 32<br/>Compact Feature Vector<br/>Encodes spatial-spectral patterns]
    end
    
    subgraph "DECODER NETWORK"
        LatentZ --> DecLayer1[Linear Layer 1<br/>Weight Matrix: W₃<br/>Shape: latent_dim × 512<br/>Bias: b₃ ∈ R^512<br/>Operation: h = zW₃ᵀ + b₃]
        
        DecLayer1 --> DecReLU1[ReLU Activation<br/>f(x) = max(0, x)<br/>Shape: batch × 512]
        
        DecReLU1 --> DecLayer2[Linear Layer 2<br/>Weight Matrix: W₄<br/>Shape: 512 × input_dim<br/>Bias: b₄ ∈ R^input_dim<br/>Operation: x̂ = hW₄ᵀ + b₄]
        
        DecLayer2 --> Reconstructed[Reconstructed Patch X̂<br/>Shape: batch × input_dim<br/>Reconstruction of Original Input]
    end
    
    subgraph "LOSS COMPUTATION"
        Input --> MSEInput
        Reconstructed --> MSEInput[MSE Loss Function<br/>L = 1/N Σᵢ ||xᵢ - x̂ᵢ||²<br/>Mean Squared Error<br/>Measures reconstruction quality]
    end
    
    subgraph "FEATURE EXTRACTION"
        LatentZ --> FeatureExtract[Latent Features Z<br/>Used for:<br/>- Anomaly Detection<br/>- Classification<br/>- Visualization]
    end
    
    subgraph "TRANSFORMER ANOMALY SCORER"
        FeatureExtract --> TransInput[Transformer Input<br/>Shape: batch × 32<br/>Latent Features Z]
        
        TransInput --> Unsqueeze[Unsqueeze Operation<br/>Add Sequence Dimension<br/>Shape: batch × 1 × 32<br/>z → z' where z'[i, 0, :] = z[i, :]]
        
        Unsqueeze --> QKV[Generate Q, K, V<br/>Query: Q = z'W_q<br/>Key: K = z'W_k<br/>Value: V = z'W_v<br/>W_q, W_k, W_v ∈ R^32×32<br/>Shape: batch × 1 × 32]
        
        QKV --> Attention[Multi-Head Self-Attention<br/>4 Attention Heads<br/>Head Dimension: 32/4 = 8<br/>Attention(Q, K, V) = softmax(QKᵀ/√d_k) · V<br/>d_k = 8 (head dimension)<br/>Output Shape: batch × 1 × 32]
        
        Attention --> AttnOut[Attention Output<br/>Shape: batch × 1 × 32<br/>Context-aware features]
        
        AttnOut --> Squeeze[Squeeze Operation<br/>Remove Sequence Dimension<br/>Shape: batch × 32<br/>z'' = squeeze(z')]
        
        Squeeze --> TransLinear1[Linear Layer 1<br/>Weight: W₅ ∈ R^32×64<br/>Bias: b₅ ∈ R^64<br/>Operation: h = z''W₅ᵀ + b₅<br/>Shape: batch × 64]
        
        TransLinear1 --> TransReLU[ReLU Activation<br/>f(x) = max(0, x)<br/>Shape: batch × 64]
        
        TransReLU --> TransLinear2[Linear Layer 2<br/>Weight: W₆ ∈ R^64×1<br/>Bias: b₆ ∈ R^1<br/>Operation: score = hW₆ᵀ + b₆<br/>Shape: batch × 1]
        
        TransLinear2 --> AnomalyScore[Anomaly Score<br/>Shape: batch × 1<br/>Higher score = Higher anomaly likelihood]
    end
    
    subgraph "ANOMALY DETECTION"
        AnomalyScore --> Normalize[Normalize Scores<br/>Min-Max Normalization<br/>score_norm = (score - min)/(max - min)<br/>Range: [0, 1]]
        
        Normalize --> AnomalyMap[Anomaly Map<br/>Spatial Distribution<br/>Shape: H × W]
        
        Reconstructed --> ReconError[Reconstruction Error<br/>Error = ||X - X̂||²<br/>MSE per sample]
        
        ReconError --> Threshold[Threshold Check<br/>threshold = 95th percentile<br/>anomaly = error > threshold]
        
        Threshold --> AnomalyMask[Anomaly Binary Mask<br/>1 = Anomaly<br/>0 = Normal]
    end
    
    %% Styling
    classDef encoderStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef decoderStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    classDef transformerStyle fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef outputStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
    classDef inputStyle fill:#fce4ec,stroke:#c2185b,stroke-width:3px
    
    class Input,Input,FeatureExtract inputStyle
    class EncLayer1,EncReLU1,EncLayer2,LatentZ encoderStyle
    class DecLayer1,DecReLU1,DecLayer2,Reconstructed decoderStyle
    class TransInput,Unsqueeze,QKV,Attention,AttnOut,Squeeze,TransLinear1,TransReLU,TransLinear2,AnomalyScore transformerStyle
    class AnomalyMap,AnomalyMask,AnomalyScore outputStyle
```

---

## Layer-by-Layer Architecture Details

### **ENCODER ARCHITECTURE**

#### Layer 1: Input Projection
```
Input Shape:      (batch, input_dim)
                  where input_dim = patch_size² × PCA_components
                  Example: (512, 4096) for 16×16 patch with 16 PCA components

Weight Matrix:    W₁ ∈ R^(input_dim × 512)
Bias Vector:      b₁ ∈ R^512
Parameters:       input_dim × 512 + 512

Operation:        h₁ = x · W₁ᵀ + b₁
Output Shape:     (batch, 512)
```

#### Activation: ReLU
```
Function:         f(x) = max(0, x)
Type:             Element-wise activation
Purpose:         Introduces non-linearity, prevents negative values
Output Shape:     (batch, 512)
```

#### Layer 2: Latent Space Compression
```
Input Shape:      (batch, 512)

Weight Matrix:    W₂ ∈ R^(512 × 32)
Bias Vector:      b₂ ∈ R^32
Parameters:       512 × 32 + 32 = 16,416

Operation:        z = ReLU(h₁) · W₂ᵀ + b₂
Output Shape:     (batch, 32) ← LATENT REPRESENTATION
```

**Total Encoder Parameters**: `input_dim × 512 + 512 + 512 × 32 + 32`

---

### **DECODER ARCHITECTURE**

#### Layer 1: Latent Expansion
```
Input Shape:      (batch, 32) ← Latent features Z

Weight Matrix:    W₃ ∈ R^(32 × 512)
Bias Vector:      b₃ ∈ R^512
Parameters:       32 × 512 + 512 = 16,896

Operation:        h₂ = z · W₃ᵀ + b₃
Output Shape:     (batch, 512)
```

#### Activation: ReLU
```
Function:         f(x) = max(0, x)
Output Shape:     (batch, 512)
```

#### Layer 2: Reconstruction Output
```
Input Shape:      (batch, 512)

Weight Matrix:    W₄ ∈ R^(512 × input_dim)
Bias Vector:      b₄ ∈ R^input_dim
Parameters:       512 × input_dim + input_dim

Operation:        x̂ = ReLU(h₂) · W₄ᵀ + b₄
Output Shape:     (batch, input_dim) ← RECONSTRUCTED PATCH
```

**Total Decoder Parameters**: `32 × 512 + 512 + 512 × input_dim + input_dim`

---

### **TRANSFORMER ANOMALY SCORER ARCHITECTURE**

#### Input Preparation
```
Input:            Latent features Z from Encoder
Shape:            (batch, 32)

Operation:        z_unsqueezed = z.unsqueeze(1)
Output Shape:     (batch, 1, 32)
Purpose:         Prepares for sequence-based attention
```

#### Multi-Head Self-Attention
```
Embedding Dim:    32
Number of Heads:  4
Head Dimension:   32 / 4 = 8

Query Matrix:     W_q ∈ R^(32 × 32), generates Q = z'W_q
Key Matrix:       W_k ∈ R^(32 × 32), generates K = z'W_k
Value Matrix:     W_v ∈ R^(32 × 32), generates V = z'W_v

Attention Mechanism:
  Q = z' · W_q    Shape: (batch, 1, 32)
  K = z' · W_k    Shape: (batch, 1, 32)
  V = z' · W_v    Shape: (batch, 1, 32)

  Attention = softmax((Q · Kᵀ) / √d_k) · V
  where d_k = 8 (head dimension)

Output Shape:     (batch, 1, 32)
```

**Attention Parameters**: `4 heads × (32 × 32) × 3 (Q, K, V) = 12,288`

#### Output Processing
```
Operation:        z_squeezed = attention_output.squeeze(1)
Output Shape:     (batch, 32)
```

#### Linear Layer 1
```
Input Shape:      (batch, 32)

Weight Matrix:    W₅ ∈ R^(32 × 64)
Bias Vector:      b₅ ∈ R^64
Parameters:       32 × 64 + 64 = 2,112

Operation:        h = z_squeezed · W₅ᵀ + b₅
Output Shape:     (batch, 64)
```

#### Activation: ReLU
```
Output Shape:     (batch, 64)
```

#### Linear Layer 2: Anomaly Score
```
Input Shape:      (batch, 64)

Weight Matrix:    W₆ ∈ R^(64 × 1)
Bias Vector:      b₆ ∈ R^1
Parameters:       64 × 1 + 1 = 65

Operation:        score = ReLU(h) · W₆ᵀ + b₆
Output Shape:     (batch, 1) ← ANOMALY SCORE
```

**Total Transformer Parameters**: `12,288 + 2,112 + 65 = 14,465`

---

## Model Parameter Summary

### Example Calculation (16×16 patch, 16 PCA components)
```
input_dim = 16 × 16 × 16 = 4,096

Encoder Parameters:
  Layer 1:  4,096 × 512 + 512 = 2,097,664
  Layer 2:  512 × 32 + 32 = 16,416
  Total:    2,114,080 parameters

Decoder Parameters:
  Layer 1:  32 × 512 + 512 = 16,896
  Layer 2:  512 × 4,096 + 4,096 = 2,100,224
  Total:    2,117,120 parameters

Autoencoder Total: 4,231,200 parameters

Transformer Parameters:
  Multi-Head Attention: 12,288
  Linear 1: 2,112
  Linear 2: 65
  Total: 14,465 parameters

Grand Total: 4,245,665 parameters
```

---

## Forward Pass Equations

### Complete Forward Pass

```
1. ENCODER:
   h₁ = ReLU(x · W₁ᵀ + b₁)
   z = h₁ · W₂ᵀ + b₂

2. DECODER:
   h₂ = ReLU(z · W₃ᵀ + b₃)
   x̂ = h₂ · W₄ᵀ + b₄

3. LOSS:
   L = (1/N) Σᵢ ||xᵢ - x̂ᵢ||²

4. TRANSFORMER:
   z' = unsqueeze(z)
   Q = z' · W_q, K = z' · W_k, V = z' · W_v
   attn_out = softmax(QKᵀ/√8) · V
   z'' = squeeze(attn_out)
   h = ReLU(z'' · W₅ᵀ + b₅)
   score = h · W₆ᵀ + b₆

5. ANOMALY DETECTION:
   error = ||x - x̂||²
   anomaly = (error > threshold_95) OR (score > threshold_score)
```

---

## Model Training Process

### Training Configuration
```
Optimizer:        Adam
Learning Rate:    0.001
Beta1 (β₁):       0.9
Beta2 (β₂):       0.999
Batch Size:       512
Epochs:           10-20
Early Stopping:   Patience = 3 epochs
Loss Function:    MSE (Mean Squared Error)
```

### Training Flow
```
1. Forward Pass:    X → Encoder → Z → Decoder → X̂
2. Loss Computation: L = MSE(X, X̂)
3. Backpropagation:  Compute gradients ∂L/∂W for all layers
4. Parameter Update: W ← W - α · ∇W (Adam optimizer)
5. Repeat until convergence or early stopping
```

---

## Model Inference Process

### Anomaly Detection Inference
```
1. Load trained model (encoder + decoder)
2. Forward pass: Extract latent features Z = Encoder(X)
3. Option A - Reconstruction Error:
   - Reconstruct: X̂ = Decoder(Z)
   - Compute: error = ||X - X̂||²
   - Anomaly if: error > threshold_95

4. Option B - Transformer Score:
   - Forward through Transformer: score = Transformer(Z)
   - Normalize: score_norm = (score - min)/(max - min)
   - Anomaly if: score_norm > threshold

5. Map scores to spatial coordinates
6. Generate anomaly heatmap
```

---

## Architecture Visualization (Text-based)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATCH AUTOENCODER MODEL                      │
└─────────────────────────────────────────────────────────────────┘

INPUT: [batch, input_dim]
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Linear(input_dim → 512)  [W₁: input_dim×512, b₁: 512]          │
│          │                                                       │
│          ▼                                                       │
│  ReLU Activation                                                 │
│          │                                                       │
│          ▼                                                       │
│  Linear(512 → 32)     [W₂: 512×32, b₂: 32]                      │
│          │                                                       │
│          ▼                                                       │
│  LATENT Z: [batch, 32]                                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
   │
   ├─────────────────────────────────────┐
   │                                     │
   ▼                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Linear(32 → 512)      [W₃: 32×512, b₃: 512]                    │
│          │                                                       │
│          ▼                                                       │
│  ReLU Activation                                                 │
│          │                                                       │
│          ▼                                                       │
│  Linear(512 → input_dim)  [W₄: 512×input_dim, b₄: input_dim]    │
│          │                                                       │
│          ▼                                                       │
│  RECONSTRUCTED: [batch, input_dim]                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SIMPLE TRANSFORMER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: Z [batch, 32]                                            │
│    │                                                             │
│    ▼                                                             │
│  Unsqueeze: [batch, 1, 32]                                       │
│    │                                                             │
│    ▼                                                             │
│  Multi-Head Attention (4 heads, dim=32)                          │
│    - Query Q = Z · W_q                                           │
│    - Key K = Z · W_k                                             │
│    - Value V = Z · W_v                                           │
│    - Attention = softmax(QKᵀ/√8) · V                             │
│    │                                                             │
│    ▼                                                             │
│  Attention Output: [batch, 1, 32]                                │
│    │                                                             │
│    ▼                                                             │
│  Squeeze: [batch, 32]                                            │
│    │                                                             │
│    ▼                                                             │
│  Linear(32 → 64)  [W₅: 32×64, b₅: 64]                            │
│    │                                                             │
│    ▼                                                             │
│  ReLU Activation                                                 │
│    │                                                             │
│    ▼                                                             │
│  Linear(64 → 1)   [W₆: 64×1, b₆: 1]                             │
│    │                                                             │
│    ▼                                                             │
│  ANOMALY SCORE: [batch, 1]                                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. **Encoder-Decoder Architecture**
- **Why**: Learn compact representations while preserving reconstruction capability
- **Latent Dimension**: 32 (balance between compression and information retention)
- **Hidden Layers**: 512 neurons (sufficient capacity for complex patterns)

### 2. **Multi-Head Self-Attention**
- **Why**: Capture relationships in latent space, identify anomalous patterns
- **Heads**: 4 (multiple perspectives on the data)
- **Self-Attention**: Patches attend to themselves (self-similarity check)

### 3. **ReLU Activations**
- **Why**: Non-linearity without gradient vanishing issues
- **Benefits**: Faster training, sparse activations

### 4. **MSE Loss**
- **Why**: Penalizes large reconstruction errors (anomalies have higher errors)
- **Benefit**: Directly relates to anomaly detection threshold

---

## Model Specifications Summary

| Component | Input Shape | Output Shape | Parameters | Purpose |
|-----------|------------|--------------|------------|---------|
| **Encoder Layer 1** | (batch, input_dim) | (batch, 512) | input_dim×512+512 | Feature extraction |
| **Encoder Layer 2** | (batch, 512) | (batch, 32) | 512×32+32 | Compression |
| **Decoder Layer 1** | (batch, 32) | (batch, 512) | 32×512+512 | Expansion |
| **Decoder Layer 2** | (batch, 512) | (batch, input_dim) | 512×input_dim+input_dim | Reconstruction |
| **Transformer Attention** | (batch, 32) | (batch, 32) | 12,288 | Context modeling |
| **Transformer Linear 1** | (batch, 32) | (batch, 64) | 2,112 | Feature transformation |
| **Transformer Linear 2** | (batch, 64) | (batch, 1) | 65 | Anomaly scoring |

---

This architecture efficiently processes hyperspectral image patches through compression, reconstruction, and anomaly scoring stages, enabling accurate anomaly detection in complex HSI datasets.

