import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    def __init__(self, config):
        self.config = config

    def load_and_process(self, data_path, annotation_path):
        """Load and preprocess data and annotations."""
        print("Loading data and annotations...")

        # Load required data columns
        required_columns = [
            's', 'fs', 'p', 'tempr',
            'Aw.1', 'Aw.2', 'Aw.3',
            'Mw.1', 'Mw.2', 'Mw.3',
            'pitch', 'roll', 'head'
        ]

        try:
            raw_data = pd.read_csv(data_path, usecols=required_columns)
            print(f"Successfully loaded data with shape: {raw_data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        try:
            annotations = pd.read_csv(annotation_path)
            print(f"Successfully loaded annotations with {len(annotations)} events")
            annotations = annotations.rename(columns={'eventStart': 'start_frame', 'eventEnd': 'end_frame'})
        except Exception as e:
            print(f"Error processing annotations: {e}")
            raise

        # Label states
        raw_data['state'] = 'Unknown'
        valid_states = {'Feeding', 'Traveling', 'Unknown', 'End'}

        for _, row in annotations.iterrows():
            if row['state'] not in valid_states:
                print(f"Warning: Unexpected state '{row['state']}' found in annotations")
                continue

            mask = (raw_data['s'] >= row['start_frame']) & (raw_data['s'] <= row['end_frame'])
            raw_data.loc[mask, 'state'] = row['state']

        return self.calculate_features(raw_data)

    def calculate_features(self, data):
        """Add calculated features to the data."""
        print("Calculating features...")
        data['norm_jerk'] = np.sqrt(data['Aw.1']**2 + data['Aw.2']**2 + data['Aw.3']**2)
        fs = data['fs'].iloc[0]
        data['depth_velocity'] = data['p'].diff() * fs
        data['pitch_rate'] = data['pitch'].diff() * fs
        data['roll_rate'] = data['roll'].diff() * fs
        data['heading_rate'] = data['head'].diff() * fs

        # Apply thresholds from config
        data['is_lunge'] = (data['depth_velocity'] > self.config['depth_velocity_threshold']).astype(int)
        data['is_high_pitch'] = (data['pitch'] > self.config['high_pitch_threshold']).astype(int)
        data['is_high_roll'] = (data['roll'].abs() > self.config['high_roll_threshold']).astype(int)

        depth_max = data['p'].max()
        deep_threshold = depth_max * 0.66
        shallow_threshold = depth_max * 0.33

        data['is_deep'] = (data['p'] >= deep_threshold).astype(int)
        data['is_shallow'] = ((data['p'] < deep_threshold) & (data['p'] >= shallow_threshold)).astype(int)
        data['temp_gradient'] = data['tempr'].diff()

        print("Feature calculation complete.")
        return data

    def split_data(self, data):
        """Split data into training and validation sets."""
        print("Splitting data into train and validation sets...")
        features = data.drop(columns=['state', 's', 'fs']).dropna()
        labels = data['state'].map({'Feeding': 1, 'Unknown': 0, 'Traveling': 2}).fillna(0).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long)
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.long)
        )

        print("Data splitting complete.")
        return train_dataset, val_dataset

if __name__ == "__main__":
    print("This script provides data processing functionality and should not be run independently.")
