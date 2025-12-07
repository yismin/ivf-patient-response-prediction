import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from dataset import IVFDataset

class ModelTrainer:
    """Train classification models for IVF patient response"""

    def __init__(self):
        self.dataset = IVFDataset()
        self.models = {}
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        """Load and preprocess dataset"""
        self.dataset.load_data()
        self.dataset.prepare_features()
        self.dataset.split_data()
        self.dataset.scale_features()
        self.dataset.save_preprocessing_objects()
        self.X_train, self.y_train = self.dataset.get_train_data()
        self.X_test, self.y_test = self.dataset.get_test_data()
        print("\nData preparation complete")

    def cross_validate(self, model, model_name):
        """Run 5-fold stratified CV and print F1-weighted scores"""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='f1_weighted')
        print(f"{model_name} CV F1-weighted: {scores.mean():.4f} ± {scores.std():.4f}")

    def train_logistic_regression(self):
        print("Training: Logistic Regression")
        model = LogisticRegression(
            multi_class='multinomial', max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.cross_validate(model, "Logistic Regression")
        model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = model
        print(f"Training accuracy: {model.score(self.X_train, self.y_train):.4f}")
        return model

    def train_random_forest(self):
        print("Training: Random Forest")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, random_state=42, class_weight='balanced', n_jobs=-1
        )
        self.cross_validate(model, "Random Forest")
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        print(f"Training accuracy: {model.score(self.X_train, self.y_train):.4f}")
        return model

    def train_xgboost(self):
        print("Training: XGBoost")
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss', use_label_encoder=False,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1
        )
        self.cross_validate(model, "XGBoost")
        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model
        print(f"Training accuracy: {model.score(self.X_train, self.y_train):.4f}")
        return model

    def train_calibrated_model(self, base_model_name='xgboost'):
        """Calibrate a model for better probability estimates"""
        print(f"Training: Calibrated {base_model_name}")
        base_model = self.models.get(base_model_name)
        if base_model is None:
            raise ValueError(f"Base model {base_model_name} not trained yet")
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(self.X_train, self.y_train)
        self.models[f'calibrated_{base_model_name}'] = model
        print(f"Training accuracy: {model.score(self.X_train, self.y_train):.4f}")
        return model

    def train_all_models(self):
        """Train all models"""
        print("TRAINING ALL MODELS")
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        # Calibrate the model that performs best on CV or your choice
        self.train_calibrated_model('xgboost')
        print(f"Trained {len(self.models)} models")
        return self.models

    def save_models(self):
        """Save all models and automatically select the true best model"""
        print("SAVING MODELS")
        for name, model in self.models.items():
            joblib.dump(model, self.model_dir / f"{name}.pkl")
            print(f"Saved: {name}.pkl")

        # Evaluate on test set
        f1_scores = {name: f1_score(self.y_test, m.predict(self.X_test), average='weighted')
                     for name, m in self.models.items()}
        best_model_name = max(f1_scores, key=f1_scores.get)
        best_model = self.models[best_model_name]
        joblib.dump(best_model, self.model_dir / "best_model.pkl")
        print(f"Saved best model: {best_model_name} (F1: {f1_scores[best_model_name]:.4f})")
        print(f"\nAll models saved to: {self.model_dir}")

    def display_probability_example(self):
        """Show example of probability outputs"""
        print("PROBABILITY OUTPUT EXAMPLE")
        best_model = joblib.load(self.model_dir / "best_model.pkl")
        sample = self.X_test[0:1]
        proba = best_model.predict_proba(sample)[0]
        prediction = best_model.predict(sample)[0]
        classes = self.dataset.label_encoder.classes_
        for cls, prob in zip(classes, proba):
            print(f"  {cls:10s}: {prob*100:.1f}%")
        predicted_class = classes[prediction]
        print(f"\nPrediction: {proba[prediction]*100:.0f}% chance this patient is {predicted_class} responsive")

def main():
    trainer = ModelTrainer()
    trainer.prepare_data()
    trainer.train_all_models()
    trainer.save_models()
    trainer.display_probability_example()
    print("TRAINING COMPLETE!")
    return trainer

if __name__ == "__main__":
    trainer = main()
