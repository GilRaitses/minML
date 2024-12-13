import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class VisualizationUtils:
    @staticmethod
    def plot_state_distribution(data, column="state", save_path=None):
        """Plot the distribution of states in the dataset."""
        state_counts = data[column].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=state_counts.index, y=state_counts.values, palette="Blues")
        plt.title("State Distribution")
        plt.xlabel("States")
        plt.ylabel("Frequency")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None):
        """Plot and optionally save a confusion matrix."""
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

    @staticmethod
    def plot_feature_correlation(data, save_path=None):
        """Plot correlation heatmap of features."""
        plt.figure(figsize=(12, 10))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', square=True)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    print("This script provides visualization utilities and should not be run independently.")
