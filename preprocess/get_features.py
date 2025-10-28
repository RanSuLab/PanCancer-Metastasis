import os
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from datetime import datetime


class MLP(nn.Module):
    """Simple MLP head for feature projection."""
    def __init__(self, input_size, common_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, common_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(common_size, common_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.linear(x)


def euclidean_distance(vec_a, vec_b):
    """Compute Euclidean distance between two vectors."""
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def normalize_adj(adj):
    """Symmetric normalization of adjacency matrix."""
    mean_val = torch.mean(adj)
    adj = (adj <= mean_val).float()

    rowsum = adj.sum(1).numpy()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.tensor(np.diag(d_inv_sqrt))

    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return adj_normalized


def cross_correlation(features):
    """
    Compute adjacency matrix based on pairwise feature distances.
    Args:
        features (torch.Tensor): [N, D] feature matrix.
    Returns:
        torch.Tensor: normalized adjacency matrix.
    """
    n = features.shape[0]
    A = torch.zeros((n, n))
    temp = A.numpy()

    for i in range(n - 1):
        for j in range(n - i - 1):
            dist = euclidean_distance(features[i].cpu().numpy(), features[i + j + 1].cpu().numpy())
            temp[i, i + j + 1] = dist
            temp[i + j + 1, i] = dist

    adj = normalize_adj(A)
    return adj


def get_features(model, img_path, graphs_path, graphs_name, batch_size, label, cluster_threshold):
    """
    Extract patch-level features and construct adjacency graphs for WSIs.
    Args:
        model (nn.Module): pretrained feature extractor.
        img_path (str): directory containing WSI patch folders.
        graphs_path (str): output directory for graph files.
        graphs_name (str): base name for saved files.
        batch_size (int): number of patches processed in a batch.
        label (int): class label (e.g., 0 or 1).
        cluster_threshold (int): number of patches for clustering decision.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(graphs_path, exist_ok=True)

    transform_ops = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    start_time = datetime.now()
    graph_data = []

    wsi_folders = [f for f in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, f))]

    for idx, folder in enumerate(wsi_folders, 1):
        print(f"[{idx}/{len(wsi_folders)}] Processing WSI: {folder}")

        folder_path = os.path.join(img_path, folder)
        patch_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")]

        patch_features = []
        patch_batch = []

        for j, patch_file in enumerate(patch_files, 1):
            img = Image.open(patch_file).convert("RGB")
            patch_tensor = transform_ops(img).unsqueeze(0)
            patch_batch.append(patch_tensor)

            if len(patch_batch) == batch_size or j == len(patch_files):
                inputs = torch.cat(patch_batch, dim=0).to(device)
                outputs = model(inputs).detach().cpu()
                patch_features.append(outputs)
                patch_batch = []

        X = torch.cat(patch_features, dim=0)

        # Optional: cluster reduction if patch count > threshold
        if X.shape[0] > cluster_threshold:
            X = X  # clustering placeholder

        adj = cross_correlation(X)
        graph_data.append({
            "WSI_name": folder,
            "flow_x": X,
            "graph": adj,
            "flow_y": label
        })

    torch.save(graph_data, os.path.join(graphs_path, f"{graphs_name}.pth"))
    elapsed = (datetime.now() - start_time).seconds
    print(f"Feature extraction completed in {elapsed}s.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract patch features and construct graphs from WSIs.")
    parser.add_argument("--img_path", type=str, required=True, help="Directory containing WSI patch folders.")
    parser.add_argument("--graphs_path", type=str, required=True, help="Directory to save graph data.")
    parser.add_argument("--graphs_name", type=str, required=True, help="Output file base name.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for feature extraction.")
    parser.add_argument("--label", type=int, required=True, help="Label for the WSIs (e.g., 0 or 1).")
    parser.add_argument("--cluster_threshold", type=int, default=300, help="Threshold for optional clustering.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()} | Device: {device}")

    densenet = models.densenet121(pretrained=True)
    densenet.classifier = MLP(input_size=1024, common_size=512)
    densenet = densenet.to(device).eval()

    get_features(
        model=densenet,
        img_path=args.img_path,
        graphs_path=args.graphs_path,
        graphs_name=args.graphs_name,
        batch_size=args.batch_size,
        label=args.label,
        cluster_threshold=args.cluster_threshold
    )
