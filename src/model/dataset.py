import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class IVFDataset:
    def __init__(self, data_path=None):
        if data_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / "data" / "processed" / "cleaned_data.csv"
        
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} patients")
        return self.df
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\nPreparing features...")
        
        # Select numerical features
        feature_cols = ['Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC', 'cycle_number']
        
        # Check which features exist
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        # One-hot encode Protocol
        protocol_dummies = pd.get_dummies(self.df['Protocol'], prefix='protocol')
        
        # Combine features
        X_numeric = self.df[feature_cols]
        self.X = pd.concat([X_numeric, protocol_dummies], axis=1)
        self.feature_names = list(self.X.columns)
        
        # Encode target labels: low=0, optimal=1, high=2
        self.y = self.label_encoder.fit_transform(self.df['Patient Response'])
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y  # Maintain class balance
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print("\nScaling features...")
        
        # Fit on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # Transform test data using training statistics
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled (StandardScaler: mean=0, std=1)")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def get_train_data(self):
        """Return training data"""
        return self.X_train_scaled, self.y_train
    
    def get_test_data(self):
        """Return test data"""
        return self.X_test_scaled, self.y_test
    
    def save_preprocessing_objects(self, output_dir=None):
        """Save scaler and label encoder for later use"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "saved_models"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        # Save label encoder
        joblib.dump(self.label_encoder, output_dir / "label_encoder.pkl")
        # Save feature names
        joblib.dump(self.feature_names, output_dir / "feature_names.pkl")
        print(f" Saved preprocessing objects to {output_dir}")    
    def load_preprocessing_objects(self, model_dir=None):
        """Load saved preprocessing objects"""
        if model_dir is None:
            model_dir = Path(__file__).parent / "saved_models"
        
        model_dir = Path(model_dir)
        
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.label_encoder = joblib.load(model_dir / "label_encoder.pkl")
        self.feature_names = joblib.load(model_dir / "feature_names.pkl")
        
        print(f" Loaded preprocessing objects from {model_dir}")
        
        return self.scaler, self.label_encoder, self.feature_names


def main():
    """Test dataset loading"""
    dataset = IVFDataset()
    dataset.load_data()
    dataset.prepare_features()
    dataset.split_data()
    dataset.scale_features()
    dataset.save_preprocessing_objects()
    
    print("DATASET READY")
    print(f"Training samples: {len(dataset.X_train_scaled)}")
    print(f"Test samples: {len(dataset.X_test_scaled)}")
    print(f"Number of features: {dataset.X_train_scaled.shape[1]}")
    print(f"Number of classes: {len(dataset.label_encoder.classes_)}")
    
    return dataset


if __name__ == "__main__":
    dataset = main()