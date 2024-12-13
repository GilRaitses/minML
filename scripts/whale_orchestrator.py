import os
import torch
import yaml
from torch.utils.data import DataLoader
from evaluate_model import evaluate, plot_confusion_matrix, compute_metrics
from gru_models import MinGRU, TraditionalGRU
from train_model import Trainer
from hmm_refinement import HMMRefiner
from rjmcmc_module import RJMCMCRefiner
from data_processing import FeatureExtractor

class WhaleOrchestrator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.annotations_dir = os.path.join(base_dir, 'annotations')
        self.config_path = os.path.join(base_dir, 'config.yaml')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Load or create config
        self.config = self.load_or_create_config()
        self.feature_extractor = FeatureExtractor(self.config)
        self.hmm_refiner = HMMRefiner(self.config)
        self.rjmcmc_refiner = RJMCMCRefiner(self.config)

    def load_or_create_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            default_config = {
                "hidden_size": 64,
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_epochs": 20,
                "use_windowing": True,
                "window_size": 25,
                "stride": 12,
                "depth_velocity_threshold": 3.4,
                "high_pitch_threshold": 30,
                "high_roll_threshold": 90
            }
            with open(self.config_path, 'w') as file:
                yaml.dump(default_config, file)
            return default_config

    def load_data(self):
        # Load and process data
        data_path = os.path.join(self.data_dir, 'data.csv')
        annotation_path = os.path.join(self.annotations_dir, 'annotations.csv')
        return self.feature_extractor.load_and_process(data_path, annotation_path)

    def train_and_evaluate(self, train_loader, val_loader):
        # Model Selection
        model_type = self.config.get("model_type", "MinGRU")
        if model_type == "MinGRU":
            model = MinGRU(self.config["hidden_size"], self.config["hidden_size"], 3)
        else:
            model = TraditionalGRU(self.config["hidden_size"], self.config["hidden_size"], 3)

        model = model.to(self.device)
        trainer = Trainer(model, train_loader, val_loader, self.config, self.device)
        trainer.train()

        print("Training complete. Evaluating model...")
        val_loss, preds, labels = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), self.device)
        metrics = compute_metrics(labels, preds)
        print(f"Validation Metrics: {metrics}")

        # Save Confusion Matrix
        plot_confusion_matrix(labels, preds, ["Unknown", "Feeding", "Traveling"],
                              save_path=os.path.join(self.base_dir, 'confusion_matrix.png'))

    def run_pipeline(self):
        print("Setting up pipeline...")
        processed_data = self.load_data()

        # Data splitting and loaders
        train_dataset, val_dataset = self.feature_extractor.split_data(processed_data)
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"])

        # Training and Evaluation
        self.train_and_evaluate(train_loader, val_loader)

        # Refinement using HMM and RJMCMC
        print("Refining states with HMM...")
        processed_data = self.hmm_refiner.refine_states(processed_data)

        print("Refining states with RJMCMC...")
        refined_data = self.rjmcmc_refiner.refine_states(processed_data)

        # Save refined data
        refined_data.to_csv(os.path.join(self.base_dir, 'refined_data.csv'), index=False)

if __name__ == "__main__":
    base_dir = input("Enter the base directory for the project: ").strip()
    orchestrator = WhaleOrchestrator(base_dir)
    orchestrator.run_pipeline()
