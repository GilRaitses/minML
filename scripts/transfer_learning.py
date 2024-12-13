import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class TransferLearning:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def fine_tune(self, new_data, new_labels):
        """Fine-tune the model on new data."""
        print("Starting fine-tuning...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            new_data, new_labels, test_size=0.2, random_state=self.config.get("random_state", 42)
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.get("batch_size", 32), shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.get("batch_size", 32)
        )

        # Freeze layers except for the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier
        self.model.fc = nn.Linear(self.model.hidden_size, len(set(new_labels)))
        self.model.fc.requires_grad = True
        self.model = self.model.to(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.fc.parameters(), lr=self.config.get("learning_rate", 0.001))
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        epochs = self.config.get("num_epochs", 20)

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.config.get("device")), labels.to(self.config.get("device"))

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.2f}%")

        print("Fine-tuning complete.")

if __name__ == "__main__":
    print("This script provides transfer learning utilities and should not be run independently.")
