# HSI Anomaly Detection Model Architecture

## Overview
This document provides a comprehensive breakdown of the Hyperspectral Imaging (HSI) anomaly detection system architecture, showing how HSI images are processed through the pipeline to identify and classify anomalies.

## System Components

### 1. Data Input Layer
- **Input Format**: MATLAB (.mat) files containing hyperspectral data cubes
- **Data Structure**: 3D tensors of shape (Height × Width × Spectral Bands)
- **Ground Truth**: Separate .mat files with pixel-level class labels

### 2. Preprocessing Pipeline
- **Noise Removal**: Dataset-specific band filtering (e.g., Indian Pines: removes bands 104-108, 150-163, 220)
- **Normalization**: MinMaxScaler normalization (values scaled to [0, 1])
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
  - Pavia: 30 components
  - Indian/Salinas: 40 components
- **Output**: Reduced HSI cube (H × W × PCA_components)

### 3. Patch Extraction Module
- **Patch Size**: Configurable (default: 16×16, classification uses 3×3 or 5×5)
- **Padding**: Reflect padding to handle boundary pixels
- **Spatial Context**: Extracts spatial-spectral patches maintaining spatial relationships
- **Output**: Flattened patches of shape (N_patches × patch_size² × PCA_components)

### 4. Autoencoder Architecture

#### Encoder Network
```
Input: Flattened Patch (patch_size² × PCA_components)
  ↓
Linear Layer: (input_dim → 512)
  ↓
ReLU Activation
  ↓
Linear Layer: (512 → latent_dim)
  ↓
Output: Latent Representation Z (latent_dim = 32)
```

#### Decoder Network
```
Input: Latent Representation Z (latent_dim = 32)
  ↓
Linear Layer: (latent_dim → 512)
  ↓
ReLU Activation
  ↓
Linear Layer: (512 → input_dim)
  ↓
Output: Reconstructed Patch (same size as input)
```

**Loss Function**: Mean Squared Error (MSE) between input and reconstructed patches
**Training**: Adam optimizer (lr=0.001), Batch size=512, Early stopping (patience=3)

### 5. Latent Feature Extraction
- Encoder outputs compact 32-dimensional feature vectors
- Captures essential spectral-spatial patterns
- Used for both anomaly detection and classification

### 6. Transformer-based Anomaly Scoring

#### SimpleTransformer Architecture
```
Input: Latent Features Z (batch × latent_dim)
  ↓
Unsqueeze: Add sequence dimension (batch × 1 × latent_dim)
  ↓
Multi-Head Self-Attention (4 heads, embed_dim=32)
  - Query: Z
  - Key: Z  
  - Value: Z
  ↓
Attention Output (batch × 1 × latent_dim)
  ↓
Squeeze: Remove sequence dimension (batch × latent_dim)
  ↓
Linear Layer 1: (latent_dim → 64)
  ↓
ReLU Activation
  ↓
Linear Layer 2: (64 → 1)
  ↓
Output: Anomaly Scores (batch × 1)
```

### 7. Classification Pipeline

#### SVM Classifier
- **Input**: PCA-reduced latent features (20 dimensions max)
- **Kernel**: RBF (Radial Basis Function)
- **Parameters**: C=5 (or 10 for classification pipeline), gamma='scale'
- **Class Balancing**: Automatic class weight computation
- **Output**: Class predictions and probabilities

### 8. Anomaly Detection Methods

#### Method 1: Reconstruction Error
- Compute MSE between input patches and reconstructed patches
- Threshold: 95th percentile of reconstruction errors
- Anomalies: Patches with error > threshold

#### Method 2: Transformer Scores
- Transformer outputs anomaly scores for each patch
- Higher scores indicate higher anomaly likelihood
- Normalized to [0, 1] range for visualization

### 9. Visualization & Output
- **Anomaly Heatmap**: Spatial distribution of anomaly scores
- **RGB Overlay**: Anomalies overlaid on PCA RGB visualization
- **Confusion Matrix**: Classification performance metrics
- **t-SNE Visualization**: 2D projection of latent space
- **Classification Reports**: Precision, Recall, F1-scores

---

## Detailed Mermaid Architecture Flowchart

```mermaid
flowchart TD
    %% Input Layer
    Start([HSI Image Input<br/>.mat file<br/>H × W × Bands]) --> LoadData[Load HSI Data Cube<br/>scipy.io.loadmat]
    GTFile([Ground Truth File<br/>.mat file<br/>Class Labels]) --> LoadGT[Load Ground Truth<br/>Pixel-level Labels]
    
    %% Preprocessing Stage
    LoadData --> CheckDataset{Dataset Type?}
    CheckDataset -->|Indian Pines| RemoveNoise[Remove Noisy Bands<br/>104-108, 150-163, 220]
    CheckDataset -->|Pavia/Salinas| Normalize
    
    RemoveNoise --> Normalize[MinMaxScaler Normalization<br/>Scale to [0, 1]<br/>Shape: H × W × Bands]
    LoadGT --> Normalize
    
    Normalize --> Reshape1[Reshape to 2D<br/>H×W, Bands]
    Reshape1 --> PCA[PCA Dimensionality Reduction<br/>Pavia: 30 components<br/>Indian/Salinas: 40 components]
    
    PCA --> Reshape2[Reshape to 3D<br/>H × W × PCA_components]
    Reshape2 --> Preprocessed[Preprocessed HSI Cube<br/>H × W × PCA_dim]
    
    %% Patch Extraction
    Preprocessed --> PatchExtract[Extract Spatial Patches<br/>Patch Size: 16×16<br/>Padding: Reflect<br/>Extract: patch_size × patch_size × PCA_dim]
    
    PatchExtract --> Flatten[Flatten Patches<br/>Shape: N_patches × patch_size² × PCA_dim<br/>Input_dim = patch_size² × PCA_dim]
    
    %% Autoencoder Training
    Flatten --> SplitData[Train/Test Split<br/>Test Size: 0.25<br/>Stratified by Labels]
    
    SplitData --> CreateTensor[Convert to PyTorch Tensor<br/>dtype: float32<br/>Shape: N × input_dim]
    
    CreateTensor --> DataLoader[Create DataLoader<br/>Batch Size: 512<br/>Shuffle: False<br/>Pin Memory: True]
    
    %% Autoencoder Architecture - Encoder
    DataLoader --> EncoderStart[ENCODER NETWORK]
    
    EncoderStart --> EncLinear1[Linear Layer 1<br/>input_dim → 512<br/>Weights: W₁ ∈ R^input_dim×512<br/>Bias: b₁ ∈ R^512]
    
    EncLinear1 --> EncReLU1[ReLU Activation<br/>max0, x]
    
    EncReLU1 --> EncLinear2[Linear Layer 2<br/>512 → latent_dim<br/>Weights: W₂ ∈ R^512×32<br/>Bias: b₂ ∈ R^32<br/>latent_dim = 32]
    
    EncLinear2 --> LatentZ[Latent Representation Z<br/>Shape: N × 32<br/>Compact Feature Vector]
    
    %% Autoencoder Architecture - Decoder
    LatentZ --> DecoderStart[DECODER NETWORK]
    
    DecoderStart --> DecLinear1[Linear Layer 1<br/>latent_dim → 512<br/>Weights: W₃ ∈ R^32×512<br/>Bias: b₃ ∈ R^512]
    
    DecLinear1 --> DecReLU1[ReLU Activation<br/>max0, x]
    
    DecReLU1 --> DecLinear2[Linear Layer 2<br/>512 → input_dim<br/>Weights: W₄ ∈ R^512×input_dim<br/>Bias: b₄ ∈ R^input_dim]
    
    DecLinear2 --> Reconstructed[Reconstructed Patch X̂<br/>Shape: N × input_dim<br/>Reconstruction of Input]
    
    %% Training Loop
    Reconstructed --> MSELoss[Compute MSE Loss<br/>L = 1/N Σ X - X̂²<br/>Mean Squared Error]
    
    MSELoss --> Backprop[Backpropagation<br/>Compute Gradients<br/>∇W, ∇b for all layers]
    
    Backprop --> Optimizer[Adam Optimizer Update<br/>Learning Rate: 0.001<br/>β₁=0.9, β₂=0.999<br/>Update: W ← W - α·∇W]
    
    Optimizer --> CheckEpoch{Epochs<br/>Completed?}
    CheckEpoch -->|No| EncoderStart
    CheckEpoch -->|Yes| EarlyStop{Early Stopping?<br/>Patience: 3}
    EarlyStop -->|No Improvement| TrainComplete
    EarlyStop -->|Continue| EncoderStart
    
    TrainComplete[Training Complete<br/>Model Saved] --> ExtractLatent
    
    %% Feature Extraction
    ExtractLatent[Extract Latent Features<br/>Z = EncoderX<br/>Shape: N × 32]
    
    %% Transformer Anomaly Scoring
    ExtractLatent --> TransInput[Transformer Input<br/>Latent Features Z<br/>Shape: batch × 32]
    
    TransInput --> TransUnsqueeze[Unsqueeze Dimension<br/>Add Sequence Dim<br/>Shape: batch × 1 × 32]
    
    TransUnsqueeze --> MultiHeadAttn[Multi-Head Self-Attention<br/>4 Heads<br/>Embed Dim: 32<br/>Query Q = Z·W_q<br/>Key K = Z·W_k<br/>Value V = Z·W_v<br/>Attention = softmaxQKᵀ/√d·V]
    
    MultiHeadAttn --> TransAttnOut[Attention Output<br/>Shape: batch × 1 × 32]
    
    TransAttnOut --> TransSqueeze[Squeeze Dimension<br/>Remove Sequence Dim<br/>Shape: batch × 32]
    
    TransSqueeze --> TransLinear1[Linear Layer 1<br/>32 → 64<br/>Weights: W₅ ∈ R^32×64<br/>Bias: b₅ ∈ R^64]
    
    TransLinear1 --> TransReLU[ReLU Activation]
    
    TransReLU --> TransLinear2[Linear Layer 2<br/>64 → 1<br/>Weights: W₆ ∈ R^64×1<br/>Bias: b₆ ∈ R^1]
    
    TransLinear2 --> AnomalyScores[Anomaly Scores<br/>Shape: batch × 1<br/>Higher = More Anomalous]
    
    %% Anomaly Map Creation
    AnomalyScores --> NormalizeScores[Normalize Scores<br/>Min-Max to [0, 1]<br/>score_norm = score - min/max - min]
    
    NormalizeScores --> CreateMap[Create Anomaly Map<br/>Map Scores to Spatial Coordinates<br/>Shape: H × W]
    
    %% Classification Branch
    ExtractLatent --> PCASVM[PCA for SVM<br/>Reduce to 20 dimensions<br/>Optional: if latent_dim > 20]
    
    PCASVM --> SVMTrain[Train SVM Classifier<br/>Kernel: RBF<br/>C: 5 or 10<br/>Gamma: 'scale'<br/>Class Weights: Balanced]
    
    SVMTrain --> SVMPredict[SVM Prediction<br/>Predict Classes<br/>Predict Probabilities]
    
    SVMPredict --> Metrics[Compute Metrics<br/>Accuracy<br/>Precision, Recall, F1<br/>Confusion Matrix]
    
    %% Reconstruction Error Method
    Reconstructed --> ReconError[Compute Reconstruction Error<br/>MSE per Sample<br/>Error = meanX - X̂²<br/>Shape: N × 1]
    
    ReconError --> CalcThreshold[Calculate Threshold<br/>95th Percentile<br/>threshold = percentileerrors, 95]
    
    CalcThreshold --> AnomalyMask[Anomaly Binary Mask<br/>anomalies = error > threshold<br/>Shape: N × 1]
    
    %% Visualization Branch
    CreateMap --> VisAnomalyMap[Anomaly Heatmap Visualization<br/>Colormap: Inferno<br/>Spatial Distribution]
    
    Preprocessed --> ExtractRGB[Extract RGB from PCA<br/>First 3 Components<br/>Normalize to [0, 1]]
    
    ExtractRGB --> VisRGB[RGB Visualization<br/>PCA Components as RGB Channels]
    
    VisAnomalyMap --> VisOverlay[Overlay Visualization<br/>RGB + Anomaly Heatmap<br/>Alpha Blending: 0.4]
    
    VisRGB --> VisOverlay
    
    ExtractLatent --> TSNE[t-SNE Visualization<br/>Reduce to 2D<br/>n_components: 2<br/>Random State: 42]
    
    TSNE --> VisLatentSpace[Latent Space Plot<br/>2D Projection<br/>Color-coded by Class]
    
    Metrics --> VisConfusion[Confusion Matrix<br/>Heatmap Visualization<br/>True vs Predicted Labels]
    
    %% Output Generation
    VisOverlay --> SaveOverlay[Save Overlay Image<br/>anomaly_map_overlay.png]
    
    VisAnomalyMap --> SaveAnomalyMap[Save Anomaly Map<br/>anomaly_map.png]
    
    VisLatentSpace --> SaveTSNE[Save t-SNE Plot<br/>tsne_visualization.png]
    
    VisConfusion --> SaveConfusion[Save Confusion Matrix<br/>confusion_matrix.png]
    
    Metrics --> SaveReport[Save Classification Report<br/>classification_report.csv<br/>Metrics: Accuracy, Precision, Recall, F1]
    
    %% Final Outputs
    SaveOverlay --> Results[Final Results]
    SaveAnomalyMap --> Results
    SaveTSNE --> Results
    SaveConfusion --> Results
    SaveReport --> Results
    AnomalyMask --> Results
    
    Results[OUTPUT RESULTS<br/>✓ Anomaly Detection Map<br/>✓ Class Predictions<br/>✓ Classification Metrics<br/>✓ Visualizations<br/>✓ Spatial Coordinates]
    
    %% Styling
    classDef inputStyle fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef preprocessStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef modelStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef decisionStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class Start,GTFile inputStyle
    class LoadData,LoadGT,Normalize,Reshape1,PCA,Reshape2,Preprocessed,PatchExtract,Flatten preprocessStyle
    class EncoderStart,EncLinear1,EncReLU1,EncLinear2,LatentZ,DecoderStart,DecLinear1,DecReLU1,DecLinear2,Reconstructed,MSELoss,Backprop,Optimizer,MultiHeadAttn,TransLinear1,TransLinear2,AnomalyScores,PCASVM,SVMTrain,SVMPredict modelStyle
    class Results,SaveOverlay,SaveAnomalyMap,SaveTSNE,SaveConfusion,SaveReport outputStyle
    class CheckDataset,CheckEpoch,EarlyStop decisionStyle
```

## Mathematical Formulations

### 1. Preprocessing
```
X_raw ∈ R^(H×W×B) → X_norm ∈ [0,1]^(H×W×B)
X_norm = (X_raw - min(X_raw)) / (max(X_raw) - min(X_raw))

X_pca = PCA(X_norm, n_components) ∈ R^(H×W×D)
where D = 30 (Pavia) or 40 (Indian/Salinas)
```

### 2. Patch Extraction
```
For each pixel (i, j):
  Patch(i,j) = X_pca[i-margin:i+margin+1, j-margin:j+margin+1, :]
  Flatten: patch ∈ R^(patch_size² × D)
  
Input dimension: input_dim = patch_size² × D
```

### 3. Encoder Forward Pass
```
h₁ = ReLU(X · W₁ + b₁)  ∈ R^512
Z = h₁ · W₂ + b₂  ∈ R^32
```

### 4. Decoder Forward Pass
```
h₂ = ReLU(Z · W₃ + b₃)  ∈ R^512
X̂ = h₂ · W₄ + b₄  ∈ R^input_dim
```

### 5. Loss Function
```
L_MSE = (1/N) Σᵢ ||Xᵢ - X̂ᵢ||²
where N is batch size
```

### 6. Transformer Attention
```
Attention(Q, K, V) = softmax(QKᵀ/√d_k) · V
where Q = Z·W_q, K = Z·W_k, V = Z·W_v, d_k = 32
```

### 7. Anomaly Score
```
Score = Linear₂(ReLU(Linear₁(Attention(Z))))
Anomaly_Map[i,j] = Score[coords⁻¹(i,j)]
```

### 8. Reconstruction Error
```
Error = (1/D) Σⱼ (X[j] - X̂[j])²
Anomaly = Error > threshold_95
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `patch_size` | 16 (default), 3-5 (classification) | Spatial context window size |
| `latent_dim` | 32 | Dimensionality of latent space |
| `encoder_hidden` | 512 | Hidden layer size in encoder/decoder |
| `pca_components` | 30 (Pavia), 40 (Indian/Salinas) | PCA dimensionality |
| `batch_size` | 512 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `num_epochs` | 10-20 | Maximum training epochs |
| `early_stopping_patience` | 3 | Early stopping patience |
| `transformer_heads` | 4 | Number of attention heads |
| `svm_C` | 5-10 | SVM regularization parameter |
| `svm_kernel` | 'rbf' | SVM kernel type |
| `anomaly_threshold` | 95th percentile | Reconstruction error threshold |

## Data Flow Dimensions

```
Input: H × W × Bands (e.g., 610 × 340 × 103 for Indian Pines)
  ↓
After PCA: H × W × D (D = 30 or 40)
  ↓
After Patch Extraction: N × patch_size × patch_size × D
  ↓
After Flattening: N × (patch_size² × D)
  ↓
Encoder Output: N × 32 (latent features)
  ↓
Decoder Output: N × (patch_size² × D) (reconstructed)
  ↓
Transformer Output: N × 1 (anomaly scores)
  ↓
Anomaly Map: H × W (spatial anomaly distribution)
```

## Performance Metrics

The system outputs:
- **Classification Accuracy**: Overall correct predictions
- **Per-Class Metrics**: Precision, Recall, F1-score for each class
- **AUC-ROC**: Area under ROC curve (if applicable)
- **Average Precision**: Macro-averaged precision
- **Confusion Matrix**: Detailed classification breakdown
- **Anomaly Detection Rate**: Percentage of anomalies identified
- **Spatial Distribution**: Visual heatmaps showing anomaly locations

