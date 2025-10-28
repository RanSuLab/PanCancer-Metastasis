import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


def adj_to_coo(adj_matrix):
    """Convert adjacency matrix to COO sparse format."""
    edge_index_temp = sp.coo_matrix(adj_matrix)
    values = edge_index_temp.data
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)
    return edge_index, values


def load_pyg_data(data_path):
    """Load preprocessed graph data saved as .npy list and convert to PyG Data objects."""
    raw_data = np.load(data_path, allow_pickle=True).tolist()
    data_list = []
    for sample in raw_data:
        x = torch.from_numpy(sample["flow_x"]) if isinstance(sample["flow_x"], np.ndarray) else sample["flow_x"]
        edge_index, edge_attr = adj_to_coo(sample["graph"])
        edge_attr = torch.tensor(edge_attr)
        y = sample["flow_y"]
        graph_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
        data_list.append(graph_data)
    return data_list


def count_labels(labels):
    """Count positive and negative samples."""
    pos = sum(labels)
    neg = len(labels) - pos
    return pos, neg


def check_dataset_overlap(train_path, test_path):
    """Check for overlapping WSI samples between train and test sets."""
    train_data = np.load(train_path, allow_pickle=True).tolist()
    test_data = np.load(test_path, allow_pickle=True).tolist()

    train_wsi = {d["WSI_name"].split("+")[0] for d in train_data}
    test_wsi = {d["WSI_name"].split("+")[0] for d in test_data}

    overlap = train_wsi & test_wsi

    print(f"Train WSI count: {len(train_wsi)} | Test WSI count: {len(test_wsi)}")
    if overlap:
        print(f"⚠️  {len(overlap)} overlapping WSIs found: {list(overlap)}")
    else:
        print("✅ No overlap between train and test sets.")
    return overlap


def plot_roc(fpr, tpr, roc_auc, epoch, save_path):
    """Plot and save ROC curve."""
    plt.figure()
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f"{save_path}/roc_epoch_{epoch}.png", dpi=300)
    plt.close()


def train_one_epoch(epoch, model, optimizer, train_data, args):
    """Train model for one epoch on a list of graph data."""
    model.train()
    total_loss, total_acc = 0.0, 0.0
    preds, trues = [], []

    baseline = 0.3
    for graph_data in train_data:
        x, y = graph_data.x, graph_data.y
        edge_index, edge_weight = graph_data.edge_index, graph_data.edge_attr
        batch = graph_data.batch

        if args.cuda:
            x, y, edge_index, edge_weight, batch = [t.cuda() for t in [x, y, edge_index, edge_weight, batch]]

        optimizer.zero_grad()
        loss_raw, _ = model.calculate_objective(x, y, edge_index, edge_weight, batch)
        loss = (loss_raw - baseline).abs() + baseline
        loss.backward()
        optimizer.step()

        correct, y_pred, _ = model.calculate_classification_error(x, y, edge_index, edge_weight, batch)
        total_loss += loss.item()
        total_acc += correct
        preds.extend(y_pred.detach().cpu().numpy())
        trues.extend(y.detach().cpu().numpy())

    metrics = {
        "loss": total_loss / len(train_data),
        "acc": total_acc / len(train_data),
        "accuracy": accuracy_score(trues, preds),
        "precision": precision_score(trues, preds, average="macro"),
        "recall": recall_score(trues, preds, average="macro"),
        "f1": f1_score(trues, preds, average="macro"),
    }

    print(
        f"Epoch {epoch} | Train Loss: {metrics['loss']:.4f} | Train Acc: {metrics['acc']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
    )
    return metrics


def validate(model, valid_loader, num_samples, args):
    """Validate model on validation data."""
    model.eval()
    valid_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, label in valid_loader:
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            loss, _ = model.calculate_objective(data, label)
            valid_loss += loss.item() * len(label)
            right, _ = model.calculate_classification_error(data, label)
            correct += right
    acc = correct / num_samples
    print(f"Validation | Loss: {valid_loss / num_samples:.4f} | Accuracy: {acc:.4f}")
    return valid_loss / num_samples, acc


def test(epoch, model, test_loader, args):
    """Evaluate model on test data and compute metrics including ROC-AUC."""
    model.eval()
    preds, trues, probs = [], [], []
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for data in test_loader:
            x, y = data.x, data.y
            edge_index, edge_weight, batch = data.edge_index, data.edge_attr, data.batch

            if args.cuda:
                x, y, edge_index, edge_weight, batch = [t.cuda() for t in [x, y, edge_index, edge_weight, batch]]

            loss, _ = model.calculate_objective(x, y, edge_index, edge_weight, batch)
            correct, pred, prob = model.calculate_classification_error(x, y, edge_index, edge_weight, batch)

            total_loss += loss.item()
            total_acc += correct
            preds.extend(pred.detach().cpu().numpy())
            trues.extend(y.detach().cpu().numpy())
            probs.extend(prob[:, 1])

    fpr, tpr, _ = roc_curve(trues, probs)
    metrics = {
        "loss": total_loss / len(test_loader),
        "acc": total_acc / len(test_loader),
        "accuracy": accuracy_score(trues, preds),
        "precision": precision_score(trues, preds, average="macro"),
        "recall": recall_score(trues, preds, average="macro"),
        "f1": f1_score(trues, preds, average="macro"),
        "auc": auc(fpr, tpr),
        "confusion_matrix": confusion_matrix(trues, preds),
        "fpr": fpr,
        "tpr": tpr,
    }

    print(
        f"Epoch {epoch} | Test Loss: {metrics['loss']:.4f} | Test Acc: {metrics['acc']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}"
    )
    return metrics


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        pt = torch.softmax(inputs, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (targets * torch.log(p)) \
               - (1 - self.alpha) * p ** self.gamma * ((1 - targets) * torch.log(1 - p))
        return loss.mean()
