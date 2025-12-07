import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class PatientPredictor:
    """Predict and explain patient IVF response."""

    def __init__(self, model_path=None):
        self.model_dir = Path(__file__).parent / "saved_models"
        
        # Load model
        if model_path is None:
            model_path = self.model_dir / "best_model.pkl"
        self.model = joblib.load(model_path)
        
        # Load preprocessing objects
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")
        self.feature_names = joblib.load(self.model_dir / "feature_names.pkl")

    def preprocess_input(self, patient_data):
        """Preprocess patient data for prediction."""
        # Convert dict to DataFrame
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])

        # Numerical features
        numerical_features = ['Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC', 'cycle_number']
        X_numeric = patient_data[numerical_features]

        # One-hot encode protocol
        protocol_cols = [col for col in self.feature_names if col.startswith('protocol_')]
        protocol_encoded = pd.DataFrame(0, index=[0], columns=protocol_cols)
        protocol = patient_data['Protocol'].values[0]
        protocol_col = f"protocol_{protocol}"
        if protocol_col in protocol_cols:
            protocol_encoded[protocol_col] = 1

        # Combine features and ensure correct order
        X = pd.concat([X_numeric.reset_index(drop=True), protocol_encoded], axis=1)
        X = X[self.feature_names]

        # Scale features
        return self.scaler.transform(X)

    def predict(self, patient_data):
        """Predict class and probability for a single patient."""
        X = self.preprocess_input(patient_data)
        pred_idx = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            'prediction': pred_class,
            'confidence': float(probs[pred_idx]),
            'probabilities': {cls: float(p) for cls, p in zip(self.label_encoder.classes_, probs)}
        }

    def predict_batch(self, patients_df):
        """Predict multiple patients."""
        results = []
        for idx, row in patients_df.iterrows():
            patient_dict = row.to_dict()
            res = self.predict(patient_dict)
            res['patient_id'] = row.get('patient_id', f'patient_{idx}')
            results.append(res)
        return pd.DataFrame(results)

    def explain_prediction(self, patient_data):
        """Explain prediction with formatted output."""
        result = self.predict(patient_data)
        print("\n--- PREDICTION ---")
        # Patient info
        if isinstance(patient_data, dict):
            print("\nPatient Information:")
            for k, v in patient_data.items():
                if k != 'patient_id':
                    print(f"  {k:15s}: {v}")

        # Main prediction sentence
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"{result['confidence']*100:.0f}% chance this patient is {result['prediction']} responsive")

        # Probability breakdown
        print("\nProbability Breakdown:")
        for cls, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {cls:10s}: {bar} {prob*100:.1f}%")

        # Simple interpretation
        conf = result['confidence']
        if conf > 0.7:
            interp = f"High confidence - patient is likely {result['prediction']} responder"
        elif conf > 0.5:
            interp = f"Moderate confidence - patient is probably {result['prediction']} responder"
        else:
            interp = "Low confidence - prediction uncertain, monitor closely"
        print("\nInterpretation:", interp)

        return result

# Examples usage
def predict_single_patient():
    """Example: predict for one patient"""
    predictor = PatientPredictor()
    patient = {
        'Age': 32,
        'AMH': 2.5,
        'n_Follicles': 12,
        'E2_day5': 450.0,
        'AFC': 15,
        'cycle_number': 1,
        'Protocol': 'flexible antagonist'
    }
    return predictor.explain_prediction(patient)

def predict_from_csv(csv_path):
    """Example: batch predictions from CSV"""
    predictor = PatientPredictor()
    df = pd.read_csv(csv_path)
    results = predictor.predict_batch(df)
    output_path = Path(csv_path).parent / "predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")
    return results

def main():
    """Run example predictions"""
    predict_single_patient()
    # Uncomment for batch prediction:
    #csv_path = "data/processed/cleaned_data.csv"
    #predict_from_csv(csv_path)

if __name__ == "__main__":
    main()
