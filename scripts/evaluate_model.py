import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, data_loader, loss_function, device):
    """Evaluate the model's performance on a given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_preds, all_labels

def plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_metrics(true_labels, predicted_labels):
    """Compute accuracy, precision, recall, and F1 score."""
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average='weighted'),
        "recall": recall_score(true_labels, predicted_labels, average='weighted'),
        "f1_score": f1_score(true_labels, predicted_labels, average='weighted')
    }
    return metrics

if __name__ == "__main__":
    print("This script provides evaluation functions and should not be run independently.")
