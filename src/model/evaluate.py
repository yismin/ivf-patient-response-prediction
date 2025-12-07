import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,classification_report, confusion_matrix,roc_auc_score, log_loss)
from sklearn.calibration import calibration_curve
from dataset import IVFDataset
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.dataset = IVFDataset()
        self.model_dir = Path(__file__).parent / "saved_models"
        self.figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare dataset"""
        self.dataset.load_data()
        self.dataset.prepare_features()
        self.dataset.split_data()
        self.dataset.scale_features()
        
        self.X_train, self.y_train = self.dataset.get_train_data()
        self.X_test, self.y_test = self.dataset.get_test_data()
        
        print("Data loaded and prepared")
        
    def load_models(self):
        """Load all trained models"""        
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl',
            'Calibrated XGBoost': 'calibrated_xgboost.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                print(f"Loaded: {name}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        
    def evaluate_model(self, name, model):
        """Evaluate a single model"""
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Multi-class AUC
        try:
            auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
        except:
            auc = None
        
        # Log loss (probability quality)
        logloss = log_loss(self.y_test, y_pred_proba)
        
        # Store results
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'log_loss': logloss,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return self.results[name]
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("MODEL EVALUATION")
        
        for name, model in self.models.items():
            print(f"{name}")
            
            results = self.evaluate_model(name, model)
            
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"F1-Score:  {results['f1']:.4f}")
            if results['auc']:
                print(f"AUC:       {results['auc']:.4f}")
            print(f"Log Loss:  {results['log_loss']:.4f}")
            
            # Classification report
            print(f"\nClassification Report:")
            print(classification_report(
                self.y_test,
                results['y_pred'],
                target_names=self.dataset.label_encoder.classes_,
                zero_division=0
            ))
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("GENERATING CONFUSION MATRICES")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            y_pred = self.results[name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=self.dataset.label_encoder.classes_,
                       yticklabels=self.dataset.label_encoder.classes_)
            
            acc = self.results[name]['accuracy']
            axes[idx].set_title(f'{name}\nAccuracy: {acc:.3f}', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Remove unused subplots
        for idx in range(len(self.models), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        save_path = self.figures_dir / "confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: confusion_matrices.png")
        plt.close()
    
    def plot_calibration_curves(self):
        """Plot calibration curves"""
        print("GENERATING CALIBRATION CURVES")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            y_pred_proba = self.results[name]['y_pred_proba']
            ax = axes[idx]
            
            # Plot calibration for each class
            for class_idx, class_name in enumerate(self.dataset.label_encoder.classes_):
                y_binary = (self.y_test == class_idx).astype(int)
                prob_class = y_pred_proba[:, class_idx]
                
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_binary, prob_class, n_bins=5, strategy='quantile'
                    )
                    ax.plot(mean_predicted_value, fraction_of_positives,
                           marker='o', label=class_name, linewidth=2)
                except:
                    pass
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.5)
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Actual Fraction')
            ax.set_title(f'{name}', fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
        
        # Remove unused subplots
        for idx in range(len(self.models), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        save_path = self.figures_dir / "calibration_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: calibration_curves.png")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("\n" + "="*60)
        print("GENERATING FEATURE IMPORTANCE")
        print("="*60)
        
        # Get tree-based models
        tree_models = {k: v for k, v in self.models.items()
                      if 'Forest' in k or 'XGBoost' in k}
        
        if not tree_models:
            print("⚠ No tree-based models found")
            return
        
        # Limit to 2 models for visualization
        tree_models = dict(list(tree_models.items())[:2])
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(14, 5))
        if len(tree_models) == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(tree_models.items()):
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                continue
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': self.dataset.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            # Plot
            ax = axes[idx]
            ax.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
            ax.set_xlabel('Importance')
            ax.set_title(f'{name}\nTop 10 Features', fontweight='bold')
            ax.invert_yaxis()
        
        plt.tight_layout()
        save_path = self.figures_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: feature_importance.png")
        plt.close()
    
    def generate_comparison_report(self):
        """Generate model comparison report"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1'] for m in self.results],
            'Log Loss': [self.results[m]['log_loss'] for m in self.results],
        })
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best model
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_f1 = comparison_df.loc[best_idx, 'F1-Score']
        
        print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
        
        # Save to CSV
        reports_dir = Path(__file__).parent.parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = reports_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ Saved comparison to: {comparison_path}")
        
        return comparison_df
    
    def medical_interpretation(self):
        """Provide medical interpretation of results"""
        print("\n" + "="*60)
        print("MEDICAL INTERPRETATION")
        print("="*60)
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        best_model = self.models[best_model_name]
        
        print(f"\nUsing best model: {best_model_name}")
        print("\nModel Performance:")
        print(f"- Accuracy: {self.results[best_model_name]['accuracy']:.1%}")
        print(f"- Can reliably predict patient response category")
        
        # Feature importance interpretation
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.dataset.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(5)
            
            print("\nTop 5 Predictive Factors:")
            for idx, row in importance_df.iterrows():
                print(f"  {idx+1}. {row['feature']}: {row['importance']:.3f}")
        
        print("\nClinical Application:")
        print("- Model provides probability estimates for each response category")
        print("- Can help clinicians adjust treatment protocols")
        print("- Enables personalized patient counseling")


def main():
    """Main evaluation pipeline"""
    
    evaluator = ModelEvaluator()
    
    # Load data and models
    evaluator.load_data()
    evaluator.load_models()
    
    # Evaluate all models
    evaluator.evaluate_all_models()
    
    # Generate visualizations
    evaluator.plot_confusion_matrices()
    evaluator.plot_calibration_curves()
    evaluator.plot_feature_importance()
    
    # Generate reports
    evaluator.generate_comparison_report()
    evaluator.medical_interpretation()
    
    print("✓ EVALUATION COMPLETE!")
    print(f"Figures saved to: {evaluator.figures_dir}")
    print(f"Reports saved to: {evaluator.figures_dir.parent}")
    
    return evaluator


if __name__ == "__main__":
    evaluator = main()