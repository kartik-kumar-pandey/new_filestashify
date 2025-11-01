import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.patches as mpatches
import scipy.io as sio

class PatchAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(PatchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
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
        return out, z  # Return both reconstruction and latent features

def _resolve_mat_key(mat_dict, preferred_key=None):
    """Return the first valid key from a loaded .mat dictionary."""
    if preferred_key and preferred_key in mat_dict:
        return preferred_key

    for key in mat_dict:
        if not key.startswith('__'):
            return key

    raise KeyError("No valid data keys found in MATLAB file")


def load_dataset(dataset_name, hsi_path, gt_path):
    hsi_mat = sio.loadmat(hsi_path)
    gt_mat = sio.loadmat(gt_path)

    hsi_key_map = {
        'pavia': 'paviaU',
        'salinas': 'salinas_corrected',
        'indian': 'indian_pines_corrected'
    }
    gt_key_map = {
        'pavia': 'paviaU_gt',
        'salinas': 'salinas_gt',
        'indian': 'indian_pines_gt'
    }

    hsi_key = _resolve_mat_key(hsi_mat, hsi_key_map.get(dataset_name))
    gt_key = _resolve_mat_key(gt_mat, gt_key_map.get(dataset_name))

    data = hsi_mat[hsi_key].astype(np.float32)
    gt = gt_mat[gt_key]

    return data, gt

def preprocess(data, gt, dataset_name):
    h, w, bands = data.shape
    if dataset_name == 'indian':
        noisy_bands = [b for b in (list(range(104, 109)) + list(range(150, 164)) + [220]) if b < bands]
        data = np.delete(data, noisy_bands, axis=2)
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, data.shape[2])
    data_scaled = scaler.fit_transform(data_reshaped)
    pca_components = 30 if dataset_name == 'pavia' else 40
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data_scaled)
    data_pca = data_pca.reshape(h, w, -1)
    data_pca = (data_pca - np.min(data_pca)) / (np.max(data_pca) - np.min(data_pca))
    return data_pca, gt, h, w, pca_components

def extract_patches(data, gt, patch_size):
    h, w, _ = data.shape
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    padded_gt = np.pad(gt, ((margin, margin), (margin, margin)), mode='reflect')
    patches, labels, coords = [], [], []
    for i in range(margin, margin + h):
        for j in range(margin, margin + w):
            # Extract patch: [i-margin:i+margin, j-margin:j+margin, :]
            # For patch_size=16, margin=8: [i-8:i+8] gives 16 elements (not 17)
            patch = padded_data[i - margin:i + margin, j - margin:j + margin, :]
            label = padded_gt[i, j]
            if label != 0:
                patches.append(patch)
                labels.append(label)
                coords.append((i - margin, j - margin))
    return np.array(patches), np.array(labels), np.array(coords), h, w

def get_pca_rgb_image(data, n_components=3):
    h, w, d = data.shape
    reshaped_data = data.reshape(-1, d)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(reshaped_data)
    transformed -= transformed.min(0)
    transformed /= transformed.max(0)
    rgb = (transformed * 255).astype(np.uint8).reshape(h, w, 3)
    return rgb

def overlay_anomalies_on_rgb(
    rgb_img, coords, y_test, anomalies, label_to_name, descriptions=None,
    title="Anomaly Classification Map (on RGB)", save_path=None
):
    h, w, _ = rgb_img.shape
    valid = [i for i, (x, y) in enumerate(coords) if 0 <= x < h and 0 <= y < w]
    coords = np.array(coords)[valid]
    y_test = np.array(y_test)[valid]
    anomalies = np.array(anomalies)[valid]
    if len(anomalies) != len(coords):
        raise ValueError(f"Length mismatch: anomalies {len(anomalies)} vs coords {len(coords)}")
    anomaly_coords = coords[anomalies]
    anomaly_labels = y_test[anomalies]
    unique_labels = np.unique(anomaly_labels)
    palette = sns.color_palette("tab20", len(unique_labels))
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    plt.figure(figsize=(14, 10))
    plt.imshow(rgb_img)
    plt.axis('off')
    for (x, y), label in zip(anomaly_coords, anomaly_labels):
        plt.scatter(y, x, s=30, color=color_map[label], edgecolors='black', linewidth=0.5)
    patches = []
    for label in unique_labels:
        name = label_to_name.get(label, f"Class {label}")
        desc = descriptions.get(name, "") if descriptions else ""
        label_text = f"{name}: {desc}" if desc else name
        patches.append(mpatches.Patch(color=color_map[label], label=label_text))
    plt.legend(
        handles=patches, loc='upper right', bbox_to_anchor=(1, 1),
        borderaxespad=0., fontsize=8, title="Legend", frameon=True
    )
    plt.title("📍 " + title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_classification_pipeline(hsi_filename, gt_filename, dataset_name, uploads_dir):
    dataset_metadata = {
        'pavia': {
            'label_to_name': {
                1: 'Asphalt',
                2: 'Meadows',
                3: 'Gravel',
                4: 'Trees',
                5: 'Painted metal sheets',
                6: 'Bare Soil',
                7: 'Bitumen',
                8: 'Self-Blocking Bricks',
                9: 'Shadows'
            },
            'descriptions': {
                'Asphalt': "Urban road surface.",
                'Meadows': "Large grassy areas.",
                'Gravel': "Loose rock surface.",
                'Trees': "Vegetation cover.",
                'Painted metal sheets': "Reflective urban structures.",
                'Bare Soil': "Dry, unplanted earth.",
                'Bitumen': "Waterproof construction surface.",
                'Self-Blocking Bricks': "Interlocking brick patterns.",
                'Shadows': "Shadowed zones in urban area."
            }
        },
        'indian': {
            'label_to_name': {
                1: 'Alfalfa',
                2: 'Corn-notill',
                3: 'Corn-mintill',
                4: 'Corn',
                5: 'Grass-pasture',
                6: 'Grass-trees',
                7: 'Grass-pasture-mowed',
                8: 'Hay-windrowed',
                9: 'Oats',
                10: 'Soybean-notill',
                11: 'Soybean-mintill',
                12: 'Soybean-clean',
                13: 'Wheat',
                14: 'Woods',
                15: 'Buildings-Grass-Trees-Drives',
                16: 'Stone-Steel-Towers'
            },
            'descriptions': {
                'Alfalfa': "Perennial flowering plant.",
                'Corn-notill': "Corn grown without tillage.",
                'Corn-mintill': "Corn with minimal tillage.",
                'Corn': "Fully cultivated corn.",
                'Grass-pasture': "Grazing grassland.",
                'Grass-trees': "Mixed vegetation.",
                'Grass-pasture-mowed': "Cut pasture field.",
                'Hay-windrowed': "Dry hay laid in rows.",
                'Oats': "Cultivated oats.",
                'Soybean-notill': "Soybean without tillage.",
                'Soybean-mintill': "Soybean with light tillage.",
                'Soybean-clean': "Soybean grown cleanly.",
                'Wheat': "Wheat crop.",
                'Woods': "Dense trees.",
                'Buildings-Grass-Trees-Drives': "Urban mix with greenery.",
                'Stone-Steel-Towers': "Tall infrastructure features."
            }
        },
        'salinas': {
            'label_to_name': {
                1: "Broccoli_green_weeds_1",
                2: "Broccoli_green_weeds_2",
                3: "Fallow",
                4: "Fallow_rough_plow",
                5: "Fallow_smooth",
                6: "Stubble",
                7: "Celery",
                8: "Grapes_untrained",
                9: "Soil_vinyard_develop",
                10: "Corn_senesced_green_weeds",
                11: "Lettuce_romaine_4wk",
                12: "Lettuce_romaine_5wk",
                13: "Lettuce_romaine_6wk",
                14: "Lettuce_romaine_7wk",
                15: "Vinyard_untrained",
                16: "Vinyard_vertical_trellis"
            },
            'descriptions': {
                "Broccoli_green_weeds_1": "Weeds in broccoli field type 1.",
                "Broccoli_green_weeds_2": "Weeds in broccoli field type 2.",
                "Fallow": "Unplanted agricultural field.",
                "Fallow_rough_plow": "Rough-plowed, uncultivated land.",
                "Fallow_smooth": "Smooth, barren agricultural field.",
                "Stubble": "Remnants of harvested crops.",
                "Celery": "Celery vegetation detected.",
                "Grapes_untrained": "Grapevines not trained on trellis.",
                "Soil_vinyard_develop": "Soil under vineyard development.",
                "Corn_senesced_green_weeds": "Dried corn with weeds.",
                "Lettuce_romaine_4wk": "4-week romaine lettuce.",
                "Lettuce_romaine_5wk": "5-week romaine lettuce.",
                "Lettuce_romaine_6wk": "6-week romaine lettuce.",
                "Lettuce_romaine_7wk": "7-week romaine lettuce.",
                "Vinyard_untrained": "Untrained vineyard vines.",
                "Vinyard_vertical_trellis": "Vineyard with vertical trellis."
            }
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hsi_path = os.path.join(uploads_dir, hsi_filename)
    gt_path = os.path.join(uploads_dir, gt_filename)
    data, gt = load_dataset(dataset_name, hsi_path, gt_path)
    data_pca, gt, h, w, pca_components = preprocess(data, gt, dataset_name)
    patch_size = 5 if dataset_name == 'indian' or dataset_name == 'salinas' else 3
    patches, labels, coords, h, w = extract_patches(data_pca, gt, patch_size)
    patches = patches.reshape(patches.shape[0], -1)
    input_dim = patches.shape[1]
    latent_dim = 32
    # First, split data before training to avoid data leakage
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        patches, labels, coords, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Convert to tensors
    patches_tensor_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    patches_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Ensure patches are flattened properly
    if len(patches_tensor_train.shape) > 2:
        patches_tensor_train = patches_tensor_train.view(patches_tensor_train.size(0), -1)
    if len(patches_tensor_test.shape) > 2:
        patches_tensor_test = patches_tensor_test.view(patches_tensor_test.size(0), -1)
    
    input_dim = patches_tensor_train.size(1)
    latent_dim = 32
    
    # Create model with consistent architecture (512 hidden units like utils.py)
    model = PatchAutoencoder(latent_dim=latent_dim, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    num_epochs = 20
    
    # Train model only once on training data
    model.train()
    dataset_train = TensorDataset(patches_tensor_train)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    
    print(f"Training autoencoder on {len(patches_tensor_train)} training patches...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader_train:
            x = batch[0]
            optimizer.zero_grad()
            x_hat, z = model(x)  # Now returns (reconstruction, latent)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
    # Extract features and compute reconstructions
    model.eval()
    with torch.no_grad():
        # Extract latent features and reconstructions
        latent_features_train = model.encoder(patches_tensor_train).cpu().numpy()
        latent_features_test = model.encoder(patches_tensor_test).cpu().numpy()
        reconstructions, _ = model(patches_tensor_test)  # Returns (reconstruction, latent)
        reconstructions = reconstructions.detach().cpu().numpy()
    
    # Train SVM classifier
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), class_weights)}
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight=class_weight_dict, probability=True)
    svm.fit(latent_features_train, y_train)
    y_pred = svm.predict(latent_features_test)
    y_proba = svm.predict_proba(latent_features_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {dataset_name.upper()}")
    confusion_matrix_path = os.path.join(uploads_dir, f'confusion_matrix_{dataset_name}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Get metadata for visualization
    label_names = dataset_metadata.get(dataset_name, {}).get('label_to_name', {})
    descriptions = dataset_metadata.get(dataset_name, {}).get('descriptions', {})
    rgb_img = get_pca_rgb_image(data_pca)
    
    # Compute reconstruction errors (ensure correct shape)
    X_test_flat = patches_tensor_test.cpu().numpy()
    reconstruction_errors = np.mean((X_test_flat - reconstructions) ** 2, axis=1)
    
    # Improved anomaly detection: combine multiple signals
    # 1. Reconstruction error (high error = unusual pattern)
    # 2. Misclassification (wrong prediction = potential anomaly)
    # 3. Low prediction confidence (uncertain predictions = potential anomaly)
    
    # Normalize reconstruction errors
    recon_error_norm = (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors) + 1e-8)
    
    # Compute prediction confidence (max probability)
    pred_confidence = np.max(y_proba, axis=1)
    
    # Misclassification indicator (1 if wrong, 0 if correct)
    misclassification = (y_pred != y_test).astype(float)
    
    # Combined anomaly score: weighted combination
    # Higher weight on misclassification and reconstruction error
    anomaly_score = 0.4 * recon_error_norm + 0.4 * misclassification + 0.2 * (1 - pred_confidence)
    
    # Use adaptive threshold (90th percentile instead of 95th for more sensitivity)
    threshold = np.percentile(anomaly_score, 90)
    anomalies = anomaly_score > threshold
    
    print(f"Anomaly Detection Summary:")
    print(f"  - Total test samples: {len(y_test)}")
    print(f"  - Detected anomalies: {np.sum(anomalies)} ({100*np.sum(anomalies)/len(anomalies):.1f}%)")
    print(f"  - Misclassifications: {np.sum(misclassification)} ({100*np.sum(misclassification)/len(misclassification):.1f}%)")
    print(f"  - Anomaly threshold: {threshold:.4f}")
    print(f"  - Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
    save_path = os.path.join(uploads_dir, f'anomaly_overlay_{dataset_name}.png')
    
    # Create t-SNE visualization
    print("Creating t-SNE visualization...")
    # Handle different scikit-learn versions (n_iter was renamed to max_iter in newer versions)
    tsne_signature = inspect.signature(TSNE.__init__)
    tsne_params = {
        'n_components': 2,
        'random_state': 42,
        'init': 'pca'
    }
    
    # Use max_iter for newer versions, n_iter for older versions
    if 'max_iter' in tsne_signature.parameters:
        tsne_params['max_iter'] = 500
    elif 'n_iter' in tsne_signature.parameters:
        tsne_params['n_iter'] = 500
    
    # learning_rate is optional - use default behavior or set to numeric value
    # 'auto' was added in scikit-learn 1.2+, but we'll use numeric for compatibility
    if 'learning_rate' in tsne_signature.parameters:
        tsne_params['learning_rate'] = 200  # Safe numeric value for all versions
    
    tsne = TSNE(**tsne_params)
    
    latent_2d = tsne.fit_transform(latent_features_test)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_test, cmap='tab20', s=5, alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"Latent Space t-SNE Visualization - {dataset_name.upper()}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    tsne_path = os.path.join(uploads_dir, f'{dataset_name}_tsne_visualization.png')
    plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save anomaly score map
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map_full = np.zeros((h, w), dtype=np.float32)
    for idx, (x, y) in enumerate(coords_test):
        if 0 <= x < h and 0 <= y < w:
            anomaly_map_full[x, y] = anomaly_score[idx]
            if anomalies[idx]:
                anomaly_map[x, y] = 1.0
    
    anomaly_map_path = os.path.join(uploads_dir, f'{dataset_name}_anomaly_score_map.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(anomaly_map_full, cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Anomaly Score')
    plt.title(f"Anomaly Score Map - {dataset_name.upper()}")
    plt.axis('off')
    plt.savefig(anomaly_map_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Overlay anomalies on RGB image
    overlay_anomalies_on_rgb(
        rgb_img, coords_test, y_test, anomalies,
        label_names, descriptions,
        title=f"Anomaly Detection Map - {dataset_name.upper()}",
        save_path=save_path
    )
    # Save classification report as CSV
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict).transpose()
    csv_path = os.path.join(uploads_dir, f'classification_report_{dataset_name}.csv')
    classification_report_df.to_csv(csv_path)
    return {
        'confusion_matrix_path': confusion_matrix_path,
        'anomaly_overlay_path': save_path,
        'classification_report_path': csv_path,
        'classification_report': classification_report_dict,
        'tsne_visualization_path': tsne_path,
        'anomaly_score_map_path': anomaly_map_path,
        'anomaly_stats': {
            'total_patches': len(y_test),
            'anomalies_detected': int(np.sum(anomalies)),
            'anomaly_percentage': float(100 * np.sum(anomalies) / len(anomalies)),
            'misclassifications': int(np.sum(misclassification)),
            'threshold': float(threshold),
            'mean_reconstruction_error': float(np.mean(reconstruction_errors))
        }
    }
