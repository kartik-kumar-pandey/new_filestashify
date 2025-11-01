import os
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight

def preprocess(data, gt, dataset_name):
    h, w, bands = data.shape
    if dataset_name == 'indian':
        noisy_bands = [b for b in (list(range(104, 109)) + list(range(150, 164)) + [220]) if b < data.shape[-1]]
        data = np.delete(data, noisy_bands, axis=2)
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, data.shape[2])
    data_scaled = scaler.fit_transform(data_reshaped).astype(np.float32)  # convert to float32 to reduce memory usage
    pca_components = 30 if dataset_name != 'indian' else 40
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_scaled)
    data_pca = data_pca.reshape(h, w, -1)
    return data_pca, gt, h, w, pca_components

def extract_patches(data, gt, patch_size):
    h, w, c = data.shape
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    padded_gt = np.pad(gt, ((margin, margin), (margin, margin)), mode='reflect')

    # Only extract patches for pixels with valid labels (label != 0)
    # This is much more memory efficient than creating all patches and filtering
    patches_list = []
    labels_list = []
    coords_list = []
    
    for i in range(margin, margin + h):
        for j in range(margin, margin + w):
            label = padded_gt[i, j]
            if label != 0:  # Only process labeled pixels
                # Extract patch: [i-margin:i+margin, j-margin:j+margin, :]
                # This gives patch_size × patch_size × channels
                # Note: i+margin is exclusive, so [i-margin:i+margin] gives margin*2 = patch_size elements
                patch = padded_data[i - margin:i + margin, j - margin:j + margin, :]
                # Verify patch shape
                if patch.shape[:2] != (patch_size, patch_size):
                    raise ValueError(f"Patch shape mismatch at ({i-margin}, {j-margin}): "
                                   f"expected ({patch_size}, {patch_size}), got {patch.shape[:2]}. "
                                   f"margin={margin}, slice=[{i-margin}:{i+margin}]")
                patches_list.append(patch)
                labels_list.append(label)
                coords_list.append((i - margin, j - margin))
    
    if len(patches_list) == 0:
        raise ValueError("No valid labeled pixels found in ground truth data")
    
    patches = np.array(patches_list, dtype=np.float32)
    labels = np.array(labels_list)
    coords = np.array(coords_list)
    
    # Ensure patches shape is correct: (num_patches, patch_size, patch_size, channels)
    # The patches are already in the correct shape from the loop extraction
    if len(patches.shape) != 4:
        raise ValueError(f"Expected patches shape (N, {patch_size}, {patch_size}, {c}), got {patches.shape}")
    
    return patches, labels, coords, h, w

class PatchAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class SimpleTransformer(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        z = z.unsqueeze(1)
        attn_out, _ = self.attn(z, z, z)
        squeezed = attn_out.squeeze(1)
        scores = self.linear(squeezed).squeeze()
        return scores

def visualize_latent_space(z, labels, dataset_name, output_dir):
    """
    Visualize the latent space using t-SNE.
    Memory-efficient: samples data if too large and uses PCA preprocessing.
    """
    # Memory optimization: use smaller sample size for better memory efficiency
    MAX_SAMPLES = 5000  # Conservative limit to avoid memory issues with t-SNE
    
    if len(z) > MAX_SAMPLES:
        print(f"Sampling {MAX_SAMPLES} from {len(z)} samples for t-SNE visualization (memory optimization)")
        # Stratified sampling to preserve label distribution
        try:
            z_sample, _, labels_sample, _ = train_test_split(
                z, labels, 
                train_size=MAX_SAMPLES, 
                stratify=labels, 
                random_state=42
            )
        except (ValueError, Exception):
            # If stratification fails, use random sampling
            indices = np.random.choice(len(z), size=min(MAX_SAMPLES, len(z)), replace=False)
            z_sample = z[indices]
            labels_sample = labels[indices]
    else:
        z_sample = z
        labels_sample = labels
    
    # Additional PCA preprocessing to reduce dimensionality before t-SNE
    # This helps with memory efficiency
    if z_sample.shape[1] > 50:
        print("Applying PCA preprocessing to reduce dimensionality before t-SNE")
        pca_pre = PCA(n_components=min(50, z_sample.shape[1]))
        z_sample = pca_pre.fit_transform(z_sample)
    
    # Calculate safe perplexity (must be < n_samples)
    n_samples = len(z_sample)
    safe_perplexity = min(30, max(5, n_samples // 4))  # Use 25% of samples as perplexity, but cap at 30
    
    tsne_signature = inspect.signature(TSNE.__init__)
    tsne_params = {
        'n_components': 2,
        'perplexity': safe_perplexity,
        'random_state': 42,
        'init': 'pca',
        'method': 'barnes_hut'  # Use Barnes-Hut approximation for memory efficiency
    }

    if 'learning_rate' in tsne_signature.parameters:
        tsne_params['learning_rate'] = 200  # Use numeric value instead of 'auto' for better compatibility

    if 'n_iter' in tsne_signature.parameters:
        tsne_params['n_iter'] = 300  # Reduced iterations for faster processing
    elif 'max_iter' in tsne_signature.parameters:
        tsne_params['max_iter'] = 300

    try:
        tsne = TSNE(**tsne_params)
    except (TypeError, ValueError) as e:
        # Fallback: remove problematic parameters
        tsne_params.pop('method', None)
        if tsne_params.get('learning_rate') == 'auto':
            tsne_params['learning_rate'] = 200
        try:
            tsne = TSNE(**tsne_params)
        except Exception as e2:
            print(f"Warning: Could not initialize t-SNE: {e2}")
            print(f"Skipping t-SNE visualization for {dataset_name}")
            return

    try:
        # Force garbage collection before t-SNE
        import gc
        gc.collect()
        
        # Try to run t-SNE with error handling
        try:
            z_2d = tsne.fit_transform(z_sample)
        except Exception as e:
            # If t-SNE fails, try with even smaller sample
            if len(z_sample) > 3000:
                print(f"t-SNE failed with {len(z_sample)} samples, trying with 3000 samples...")
                # Further reduce sample size
                indices = np.random.choice(len(z_sample), size=3000, replace=False)
                z_sample_small = z_sample[indices]
                labels_sample_small = labels_sample[indices]
                
                # Adjust perplexity for smaller sample
                safe_perplexity = min(30, max(5, len(z_sample_small) // 4))
                
                # Recreate tsne_params with safe values
                tsne_params_small = {
                    'n_components': 2,
                    'perplexity': safe_perplexity,
                    'random_state': 42,
                    'init': 'pca',
                    'method': 'barnes_hut'
                }
                
                if 'learning_rate' in inspect.signature(TSNE.__init__).parameters:
                    tsne_params_small['learning_rate'] = 200
                    
                if 'max_iter' in inspect.signature(TSNE.__init__).parameters:
                    tsne_params_small['max_iter'] = 300
                elif 'n_iter' in inspect.signature(TSNE.__init__).parameters:
                    tsne_params_small['n_iter'] = 300
                
                try:
                    tsne_small = TSNE(**tsne_params_small)
                except (TypeError, ValueError):
                    tsne_params_small.pop('method', None)
                    tsne_small = TSNE(**tsne_params_small)
                
                z_2d = tsne_small.fit_transform(z_sample_small)
                labels_sample = labels_sample_small
            else:
                raise
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_sample, cmap='tab20', s=5, alpha=0.6)
        title = f"Latent Space t-SNE Visualization - {dataset_name.upper()}"
        if len(z_sample) < len(z) if hasattr(z, '__len__') else False:
            title += f"\n(Sampled: {len(z_sample):,}/{len(z):,} points)"
        plt.title(title)
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataset_name}_tsne_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clean up memory
        del z_2d, z_sample, labels_sample, tsne
        gc.collect()
        
    except BaseException as e:  # Catch all exceptions including MemoryError from C extensions
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Warning: t-SNE visualization failed: {error_type}: {error_msg}")
        print(f"Skipping t-SNE visualization for {dataset_name} to avoid memory issues")
        # Skip visualization gracefully without crashing the pipeline
        import gc
        gc.collect()

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def run_pipeline_with_files(hsi_path, gt_path, dataset_name, patch_size=16, latent_dim=32, num_epochs=10, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(hsi_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sio.loadmat(hsi_path)
    gt = sio.loadmat(gt_path)

    data_keys = [key for key in data.keys() if not key.startswith('__')]
    gt_keys = [key for key in gt.keys() if not key.startswith('__')]

    if not data_keys or not gt_keys:
        raise ValueError("No valid variable keys found in .mat files")

    data_array = data[data_keys[0]]
    gt_array = gt[gt_keys[0]]

    print("Loading dataset and preprocessing...")
    data_pca, gt_processed, h, w, pca_dim = preprocess(data_array, gt_array, dataset_name)

    rgb_image = data_pca[:, :, :3]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image_path = os.path.join(output_dir, f"{dataset_name}_pca_rgb.png")
    plt.imsave(rgb_image_path, rgb_image)

    input_dim = patch_size * patch_size * pca_dim
    patches, labels, coords, h, w = extract_patches(data_pca, gt_processed, patch_size=patch_size)

    print(f"Extracted {len(patches)} patches with known labels (no redundancy).")

    # Flatten patches: (N, patch_size, patch_size, channels) -> (N, patch_size * patch_size * channels)
    patches_flat = patches.reshape(len(patches), -1)
    
    # Verify the flattened shape matches expected input_dim
    actual_dim = patches_flat.shape[1]
    if actual_dim != input_dim:
        raise ValueError(f"Patch dimension mismatch: expected {input_dim}, got {actual_dim}. "
                        f"Patch shape: {patches.shape}, patch_size: {patch_size}, pca_dim: {pca_dim}")
    
    patches_tensor = torch.tensor(patches_flat, dtype=torch.float32)
    dataset = TensorDataset(patches_tensor)
    batch_size = 512
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=3)

    print("Training Autoencoder with mixed precision and batching...")
    model.train()
    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output, _ = model(batch)
                loss = criterion(output, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")

    loss_curve_path = os.path.join(output_dir, f"{dataset_name}_ae_loss_curve.png")
    plt.figure()
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(loss_curve_path)
    plt.close()

    print("Extracting latent features...")
    model.eval()
    with torch.no_grad():
        _, latent_z = model(patches_tensor.to(device))
    latent_z = latent_z.cpu()

    print("Visualizing latent space with optimized t-SNE...")
    visualize_latent_space(latent_z.numpy(), labels, dataset_name, output_dir)

    print("Running Transformer with batched scoring...")
    transformer = SimpleTransformer(dim=latent_dim).to(device)
    transformer.eval()
    trans_scores = []
    for i in range(0, latent_z.shape[0], batch_size):
        batch = latent_z[i:i+batch_size].to(device)
        with torch.no_grad():
            scores = transformer(batch).cpu().numpy()
        trans_scores.append(scores)
    trans_scores = np.concatenate(trans_scores)

    print("Creating anomaly map...")
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    for idx, (x, y) in enumerate(coords):
        anomaly_map[x, y] = trans_scores[idx]
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    rgb_image = data_pca[:, :, :3]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    axs[0].imshow(gt_processed, cmap='tab20')
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(rgb_image)
    axs[1].set_title("RGB PCA Image")
    axs[1].axis('off')

    im = axs[2].imshow(anomaly_map_norm, cmap='inferno')
    axs[2].set_title("Anomaly Heatmap")
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    axs[3].imshow(rgb_image)
    axs[3].imshow(anomaly_map_norm, cmap='inferno', alpha=0.4)
    axs[3].set_title("Overlay RGB + Anomalies")
    axs[3].axis('off')

    plt.tight_layout()
    overlay_path = os.path.join(output_dir, f"{dataset_name}_anomaly_map_overlay.png")
    plt.savefig(overlay_path)
    plt.close()

    # Save anomaly score map as image
    anomaly_map_path = os.path.join(output_dir, f"{dataset_name}_anomaly_map.png")
    plt.imsave(anomaly_map_path, anomaly_map_norm, cmap='inferno')

    print("Training SVM on PCA-reduced latent features...")
    pca_svm = PCA(n_components=min(latent_dim, 20))
    latent_reduced = pca_svm.fit_transform(latent_z.numpy())

    X_train, X_test, y_train, y_test = train_test_split(latent_reduced, labels, test_size=0.25, random_state=42, stratify=labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), class_weights)}

    svm_clf = SVC(kernel='rbf', C=5, gamma='scale', class_weight=class_weight_dict, probability=True)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_proba = svm_clf.predict_proba(X_test)

    classification_report_str = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = None
    ap = None
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        ap = average_precision_score(y_test, y_proba, average='macro')
    except Exception as e:
        classification_report_str += f"\nAUC/AP metrics failed: {str(e)}"

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {dataset_name.upper()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    conf_matrix_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()

    print("Pipeline complete! Results saved in output directory.")

    results = {
        'stats': {
            'accuracy': accuracy,
            'classification_report': classification_report_str,
            'auc': auc,
            'average_precision': ap
        },
        'images': [
            {
                'url': f'/uploads/{dataset_name}_confusion_matrix.png',
                'name': 'Confusion Matrix',
                'description': 'Visualization of model predictions vs true labels'
            },
            {
                'url': f'/uploads/{dataset_name}_tsne_visualization.png',
                'name': 't-SNE Visualization',
                'description': '2D visualization of the latent space'
            },
            {
                'url': f'/uploads/{dataset_name}_anomaly_map.png',
                'name': 'Anomaly Score Map',
                'description': 'Spatial distribution of anomaly scores'
            },
            {
                'url': f'/uploads/{dataset_name}_pca_rgb.png',
                'name': 'PCA RGB Image',
                'description': 'RGB image from PCA components'
            },
            {
                'url': f'/uploads/{dataset_name}_ae_loss_curve.png',
                'name': 'Autoencoder Loss Curve',
                'description': 'Training loss curve of the autoencoder'
            },
            {
                'url': f'/uploads/{dataset_name}_anomaly_map_overlay.png',
                'name': 'Anomaly Map Overlay',
                'description': 'Overlay of RGB image and anomaly heatmap'
            }
        ],
        'info': f'Analysis completed for {dataset_name} dataset with {len(labels)} samples'
    }

    return results

