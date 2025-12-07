"""
Unit tests for IVF Patient Response Prediction Model
Tests data loading, preprocessing, model prediction, and API endpoints
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Add src directories to path
sys.path.append(str(Path(__file__).parent.parent / "src" / "model"))
sys.path.append(str(Path(__file__).parent.parent / "src" / "preprocessing"))

from dataset import IVFDataset
from predict import PatientPredictor
from clean_dataset import DataCleaner


class TestDataset(unittest.TestCase):
    """Test dataset loading and preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dataset = IVFDataset()
    
    def test_data_loading(self):
        """Test that data loads correctly"""
        df = self.dataset.load_data()
        
        self.assertIsNotNone(df, "Dataset should not be None")
        self.assertGreater(len(df), 0, "Dataset should have rows")
        self.assertIn('Patient Response', df.columns, "Should have target column")
    
    def test_feature_preparation(self):
        """Test feature preparation"""
        self.dataset.load_data()
        X, y = self.dataset.prepare_features()
        
        self.assertIsNotNone(X, "Features should not be None")
        self.assertIsNotNone(y, "Target should not be None")
        self.assertEqual(len(X), len(y), "Features and target should have same length")
        self.assertGreater(X.shape[1], 5, "Should have multiple features")
    
    def test_data_splitting(self):
        """Test train/test split"""
        self.dataset.load_data()
        self.dataset.prepare_features()
        X_train, X_test, y_train, y_test = self.dataset.split_data()
        
        self.assertGreater(len(X_train), len(X_test), "Train set should be larger")
        self.assertGreater(len(X_train), 0, "Train set should not be empty")
        self.assertGreater(len(X_test), 0, "Test set should not be empty")
    
    def test_feature_scaling(self):
        """Test that features are properly scaled"""
        self.dataset.load_data()
        self.dataset.prepare_features()
        self.dataset.split_data()
        X_train_scaled, X_test_scaled = self.dataset.scale_features()
        
        # Check that scaling was applied (mean should be close to 0, std close to 1)
        train_mean = np.mean(X_train_scaled)
        train_std = np.std(X_train_scaled)
        
        self.assertAlmostEqual(train_mean, 0.0, delta=0.5, 
                             msg="Scaled features should have mean near 0")
        self.assertAlmostEqual(train_std, 1.0, delta=0.5,
                             msg="Scaled features should have std near 1")


class TestPredictor(unittest.TestCase):
    """Test prediction functionality"""
    
    def setUp(self):
        """Set up predictor"""
        self.predictor = PatientPredictor()
        
        # Sample patient data
        self.sample_patient = {
            'Age': 32,
            'AMH': 2.5,
            'n_Follicles': 12,
            'E2_day5': 450.0,
            'AFC': 15,
            'cycle_number': 1,
            'Protocol': 'flexible antagonist'
        }
    
    def test_model_loaded(self):
        """Test that model is loaded"""
        self.assertIsNotNone(self.predictor.model, "Model should be loaded")
        self.assertIsNotNone(self.predictor.scaler, "Scaler should be loaded")
        self.assertIsNotNone(self.predictor.label_encoder, "Label encoder should be loaded")
    
    def test_prediction_output(self):
        """Test prediction returns correct format"""
        result = self.predictor.predict(self.sample_patient)
        
        self.assertIn('prediction', result, "Should have prediction key")
        self.assertIn('confidence', result, "Should have confidence key")
        self.assertIn('probabilities', result, "Should have probabilities key")
        
        # Check prediction is valid class
        valid_classes = ['low', 'optimal', 'high']
        self.assertIn(result['prediction'], valid_classes, 
                     "Prediction should be valid class")
        
        # Check confidence is probability
        self.assertGreaterEqual(result['confidence'], 0.0, "Confidence should be >= 0")
        self.assertLessEqual(result['confidence'], 1.0, "Confidence should be <= 1")
        
        # Check probabilities sum to 1
        prob_sum = sum(result['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=5,
                             msg="Probabilities should sum to 1")
    
    def test_preprocessing_pipeline(self):
        """Test that preprocessing works correctly"""
        X = self.predictor.preprocess_input(self.sample_patient)
        
        self.assertIsNotNone(X, "Preprocessed input should not be None")
        self.assertEqual(len(X.shape), 2, "Should be 2D array")
        self.assertEqual(X.shape[0], 1, "Should have 1 sample")
        self.assertEqual(X.shape[1], len(self.predictor.feature_names),
                        "Should have correct number of features")
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        # Create multiple patients
        patients_df = pd.DataFrame([
            self.sample_patient,
            {**self.sample_patient, 'Age': 35, 'AMH': 1.5},
            {**self.sample_patient, 'Age': 28, 'AMH': 4.0}
        ])
        
        results = self.predictor.predict_batch(patients_df)
        
        self.assertEqual(len(results), 3, "Should predict for all patients")
        self.assertIn('prediction', results.columns, "Should have prediction column")


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functionality"""
    
    def setUp(self):
        """Create sample dirty data"""
        self.sample_data = pd.DataFrame({
            'patient_id': ['John Doe', 'Jane Smith', '25003'],
            'Age': [32, 35, 28],
            'AMH': [2.5, np.nan, 3.0],
            'n_Follicles': [12, 10, 15],
            'E2_day5': [450.0, 300.0, 550.0],
            'AFC': [15, np.nan, 18],
            'cycle_number': [1, 2, 1],
            'Protocol': ['Flex anta', 'fix antag', 'agonist'],
            'Patient Response': ['optimal', 'low', 'high']
        })
    
    def test_deidentification(self):
        """Test patient name de-identification"""
        cleaner = DataCleaner(self.sample_data)
        mapping = cleaner.de_identify_patients()
        
        # Check that names were converted to 25XXX format
        for patient_id in cleaner.df['patient_id']:
            self.assertTrue(str(patient_id).startswith('25'), 
                          f"Patient ID {patient_id} should start with 25")
    
    def test_protocol_standardization(self):
        """Test protocol standardization"""
        cleaner = DataCleaner(self.sample_data)
        cleaner.standardize_protocols()
        
        valid_protocols = ['fixed antagonist', 'flexible antagonist', 'agonist']
        for protocol in cleaner.df['Protocol']:
            self.assertIn(protocol, valid_protocols,
                        f"Protocol {protocol} should be standardized")
    
    def test_missing_value_handling(self):
        """Test that missing values are handled"""
        cleaner = DataCleaner(self.sample_data)
        cleaner.handle_missing_values()
        
        # Check that critical columns have no missing values
        critical_cols = ['Age', 'AMH', 'n_Follicles', 'AFC']
        for col in critical_cols:
            if col in cleaner.df.columns:
                missing = cleaner.df[col].isnull().sum()
                self.assertEqual(missing, 0, 
                               f"Column {col} should have no missing values")


class TestModelPerformance(unittest.TestCase):
    """Test model performance meets minimum requirements"""
    
    def setUp(self):
        """Load model and test data"""
        model_dir = Path(__file__).parent.parent / "src" / "model" / "saved_models"
        self.model = joblib.load(model_dir / "best_model.pkl")
        
        # Load test data
        self.dataset = IVFDataset()
        self.dataset.load_data()
        self.dataset.prepare_features()
        self.dataset.split_data()
        self.dataset.scale_features()
        self.X_test, self.y_test = self.dataset.get_test_data()
    
    def test_minimum_accuracy(self):
        """Test that model meets minimum accuracy requirement"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Require at least 70% accuracy (adjust based on your requirements)
        self.assertGreaterEqual(accuracy, 0.70,
                              f"Model accuracy {accuracy:.3f} should be >= 70%")
    
    def test_probability_calibration(self):
        """Test that model outputs valid probabilities"""
        y_proba = self.model.predict_proba(self.X_test)
        
        # Check all probabilities are between 0 and 1
        self.assertTrue(np.all(y_proba >= 0), "Probabilities should be >= 0")
        self.assertTrue(np.all(y_proba <= 1), "Probabilities should be <= 1")
        
        # Check probabilities sum to 1 for each sample
        prob_sums = np.sum(y_proba, axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)),
                                            decimal=5,
                                            err_msg="Probabilities should sum to 1")


class TestAPICompatibility(unittest.TestCase):
    """Test API input/output compatibility"""
    
    def test_api_input_format(self):
        """Test that predictor accepts API format input"""
        predictor = PatientPredictor()
        
        # Simulate API request format
        api_input = {
            'Age': 32,
            'AMH': 2.5,
            'n_Follicles': 12,
            'E2_day5': 450.0,
            'AFC': 15,
            'cycle_number': 1,
            'Protocol': 'flexible antagonist'
        }
        
        try:
            result = predictor.predict(api_input)
            success = True
        except Exception as e:
            success = False
            print(f"API compatibility error: {e}")
        
        self.assertTrue(success, "Predictor should accept API format input")
    
    def test_invalid_protocol_handling(self):
        """Test handling of invalid protocol values"""
        predictor = PatientPredictor()
        
        invalid_input = {
            'Age': 32,
            'AMH': 2.5,
            'n_Follicles': 12,
            'E2_day5': 450.0,
            'AFC': 15,
            'cycle_number': 1,
            'Protocol': 'invalid_protocol'  # Invalid
        }
        
        # Should handle gracefully (either error or default)
        try:
            result = predictor.predict(invalid_input)
            # If it doesn't raise error, check it still returns valid output
            self.assertIn('prediction', result)
        except (ValueError, KeyError):
            # Expected to raise error for invalid protocol
            pass


def run_all_tests():
    """Run all tests and generate report"""
    
    print("\n" + "="*60)
    print("RUNNING MODEL TESTS")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCleaning))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestAPICompatibility))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n⚠ SOME TESTS FAILED")
    
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    run_all_tests() 