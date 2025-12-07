import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import IVFDataset
import warnings
warnings.filterwarnings('ignore')

def explain_model():
    """
    Main function - explains the ML model step by step
    """
    print("STEP 1: LOADING MODEL AND DATA")
    model_dir = Path(__file__).parent / "saved_models"
    figures_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Load the trained model
    model = joblib.load(model_dir / "random_forest.pkl")
    feature_names = joblib.load(model_dir / "feature_names.pkl")
    print(f"Loaded Random Forest model for explanation")
    print(f"Model uses {len(feature_names)} features: {feature_names[:5]}...")    
    # We need the model to explain its predictions!
    print("STEP 2: LOADING TEST DATA")    
    # Load the same data we used for testing
    dataset = IVFDataset()
    dataset.load_data()
    dataset.prepare_features()
    dataset.split_data()
    dataset.scale_features()
    
    X_test, y_test = dataset.get_test_data()    
    # WHY? We want to explain predictions on data the model hasn't seen during training
    print("STEP 3: CREATING SHAP EXPLAINER")    
    explainer = shap.TreeExplainer(model)    
    # The explainer figures out how each feature affects predictions
    print("STEP 4: CALCULATING SHAP VALUES FOR 100 PATIENTS")    
    # Calculate SHAP values for first 100 patients 
    # SHAP values = how much each feature pushed the prediction up or down
    shap_values = explainer.shap_values(X_test[:100])
    # WHAT ARE SHAP VALUES?
    # Example: For Patient 1:
    #   - AMH: +0.3 (pushes prediction toward "high response")
    #   - Age: -0.1 (pushes prediction toward "low response")
    #   - AFC: +0.2 (pushes prediction toward "high response")

    print("STEP 5: CREATING VISUALIZATIONS")
    # PLOT 1: FEATURE IMPORTANCE (Bar Chart)    
    plt.figure(figsize=(10, 6))
    # For multi-class, shap_values is a list [class0, class1, class2]
    # We need to use absolute values across all classes
    if isinstance(shap_values, list):
        # Stack all classes and take mean absolute values
        all_shap = np.abs(np.array(shap_values)).mean(axis=0)
        mean_abs_shap = np.abs(all_shap).mean(axis=0)
        # Create manual bar plot
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('|SHAP value|')
        plt.title('Which Features Matter Most?', fontsize=14, fontweight='bold')
    else:
        # Single class case
        shap.summary_plot(
            shap_values,
            X_test[:100],
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title("Which Features Matter Most?", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_feature_importance.png", dpi=300, bbox_inches='tight')
    print("  Saved: shap_feature_importance.png")
    plt.close()
    # WHAT THIS SHOWS:
    # - Longer bars = more important features
    # - Example: If AMH has the longest bar, it's the most important for predictions

    # PLOT 2: DETAILED FEATURE EFFECTS (Beeswarm/Dot Plot)    
    plt.figure(figsize=(10, 8))
    
    # For multi-class models, show one class at a time
    # Let's explain "optimal" response (most common)
    class_names = dataset.label_encoder.classes_
    optimal_idx = list(class_names).index('optimal')
    
    # Use SHAP values for optimal class only
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[optimal_idx]
    else:
        shap_values_to_plot = shap_values
    
    # This shows HOW feature values affect predictions
    shap.summary_plot(
        shap_values_to_plot,
        X_test[:100],
        feature_names=feature_names,
        show=False
    )
    
    plt.title("How Do Feature Values Affect 'Optimal' Predictions?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_detailed_effects.png", dpi=300, bbox_inches='tight')
    print("   Saved: shap_detailed_effects.png")
    plt.close()
    
    # WHAT THIS SHOWS:
    # - Each dot = one patient
    # - RED dots = high feature value (e.g., high AMH)
    # - BLUE dots = low feature value (e.g., low AMH)
    # - Position LEFT = pushes toward "low response"
    # - Position RIGHT = pushes toward "high response"
    #
    # Example: If red dots (high AMH) are on the right:
    # → High AMH increases chance of optimal/high response 
    
    # PLOT 3: SINGLE PATIENT EXPLANATION (Waterfall)
    # Pick one patient to explain
    patient_idx = 0
    
    # Get their prediction
    prediction = model.predict(X_test[patient_idx:patient_idx+1])[0]
    predicted_class = class_names[prediction]
    
    # Get SHAP values for this patient
    patient_shap_values = shap_values[prediction][patient_idx]
    base_value = explainer.expected_value[prediction]
    
    # Create explanation
    explanation = shap.Explanation(
        values=patient_shap_values,
        base_values=base_value,
        data=X_test[patient_idx],
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"Why Was This Patient Predicted as '{predicted_class.upper()}'?", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_single_patient.png", dpi=300, bbox_inches='tight')
    print("  Saved: shap_single_patient.png")
    plt.close()
    
    # WHAT THIS SHOWS:
    # - Starts with "base value" (average prediction)
    # - Each bar shows how one feature pushes the prediction
    # - RED bars = push UP (toward high response)
    # - BLUE bars = push DOWN (toward low response)
    # - Final value at top = actual prediction
    #
    # Example waterfall:
    # Base value: 0.5
    # + AMH: +0.2    (good AMH pushes up)
    # + AFC: +0.15   (good AFC pushes up)
    # - Age: -0.1    (older age pushes down)
    # = Final: 0.75  (75% chance optimal)
    
    print(f"\nSaved to: {figures_dir}")
    return explainer, shap_values

if __name__ == "__main__":
    explainer, shap_values = explain_model()