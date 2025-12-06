"""
Data Cleaning and Preprocessing for IVF Patient Dataset
Handles de-identification, missing values, standardization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


class DataCleaner:
    """Clean and preprocess IVF patient data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_report = []
        
    def de_identify_patients(self):
        """Convert patient names to 25XXX format"""
        
        print("De-identifying patient names...")
        
        # Check if patient_id contains names (strings with spaces or letters beyond numbers)
        needs_deidentification = self.df['patient_id'].astype(str).str.contains('[A-Za-z]', na=False)
        
        if needs_deidentification.any():
            # Create mapping dictionary
            patient_mapping = {}
            
            for idx, patient_name in enumerate(self.df['patient_id'].unique(), start=1):
                if isinstance(patient_name, str) and re.search('[A-Za-z]', patient_name):
                    patient_mapping[patient_name] = f"25{idx:03d}"
            
            # Apply mapping
            self.df['patient_id'] = self.df['patient_id'].map(
                lambda x: patient_mapping.get(x, x)
            )
            
            self.cleaning_report.append(
                f"✓ De-identified {len(patient_mapping)} patient names"
            )
            
            return patient_mapping
        else:
            self.cleaning_report.append("✓ Patient IDs already de-identified")
            return {}
    
    def standardize_protocols(self):
        """Standardize protocol naming"""
        
        print("Standardizing protocol names...")
        
        # Comprehensive protocol mapping
        protocol_mapping = {
            # Fixed antagonist variations
            'fix antag': 'fixed antagonist',
            'Fix antag': 'fixed antagonist',
            'fixed antag': 'fixed antagonist',
            'fix anta': 'fixed antagonist',
            'Fix anta': 'fixed antagonist',
            'fixed anta': 'fixed antagonist',
            'fixed antagonist': 'fixed antagonist',
            
            # Flexible antagonist variations
            'flex anta': 'flexible antagonist',
            'Flex anta': 'flexible antagonist',
            'Flex Antago': 'flexible antagonist',
            'flexible antag': 'flexible antagonist',
            'flex antag': 'flexible antagonist',
            'Flex antag': 'flexible antagonist',
            'flexible antagonist': 'flexible antagonist',
            
            # Agonist variations
            'Agonist': 'agonist',
            'agonist': 'agonist',
            'agoni': 'agonist',
            'Agoni': 'agonist',
        }
        
        # Apply mapping
        self.df['Protocol'] = self.df['Protocol'].replace(protocol_mapping)
        
        # Check for any unmapped protocols
        valid_protocols = ['fixed antagonist', 'flexible antagonist', 'agonist']
        invalid = ~self.df['Protocol'].isin(valid_protocols)
        
        if invalid.any():
            print(f"⚠ Warning: Found {invalid.sum()} invalid protocols:")
            print(self.df.loc[invalid, 'Protocol'].unique())
            
            # Try fuzzy matching for remaining
            for idx in self.df[invalid].index:
                protocol = str(self.df.loc[idx, 'Protocol']).lower()
                if 'fix' in protocol or 'fixed' in protocol:
                    self.df.loc[idx, 'Protocol'] = 'fixed antagonist'
                elif 'flex' in protocol or 'flexible' in protocol:
                    self.df.loc[idx, 'Protocol'] = 'flexible antagonist'
                elif 'agon' in protocol:
                    self.df.loc[idx, 'Protocol'] = 'agonist'
        
        final_protocols = self.df['Protocol'].unique()
        self.cleaning_report.append(
            f"✓ Standardized protocols to: {list(final_protocols)}"
        )
    
    def handle_missing_values(self):
        """Handle missing values with appropriate strategies"""
        
        print("\nHandling missing values...")
        
        missing_before = self.df.isnull().sum()
        print("\nMissing data before imputation:")
        for col, count in missing_before[missing_before > 0].items():
            pct = (count / len(self.df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
        
        # STRATEGY 1: AFC - Use KNN imputation (better for high missingness)
        if 'AFC' in self.df.columns:
            afc_missing = self.df['AFC'].isnull().sum()
            if afc_missing > 0:
                afc_missing_pct = (afc_missing / len(self.df)) * 100
                
                if afc_missing_pct > 30:  # High missingness
                    print(f"\n⚠ AFC has {afc_missing_pct:.1f}% missing - using KNN imputation")
                    
                    # Use KNN imputation based on correlated features
                    impute_features = ['Age', 'AMH', 'n_Follicles', 'AFC', 'E2_day5']
                    impute_features = [f for f in impute_features if f in self.df.columns]
                    
                    # Create imputer
                    imputer = KNNImputer(n_neighbors=5, weights='distance')
                    
                    # Impute
                    imputed_data = imputer.fit_transform(self.df[impute_features])
                    self.df[impute_features] = imputed_data
                    
                    # AFC should be INTEGER (discrete follicle count)
                    self.df['AFC'] = self.df['AFC'].round().astype(int)
                    
                    self.cleaning_report.append(
                        f"✓ Used KNN imputation for AFC ({afc_missing} values, {afc_missing_pct:.1f}%) - rounded to integers"
                    )
                else:
                    # Low missingness - use median by age group
                    self.df['age_group'] = pd.cut(
                        self.df['Age'], 
                        bins=[0, 30, 35, 50], 
                        labels=['<30', '30-35', '>35']
                    )
                    self.df['AFC'] = self.df.groupby('age_group')['AFC'].transform(
                        lambda x: x.fillna(x.median())
                    )
                    self.df['AFC'] = self.df['AFC'].round().astype(int)
                    self.df.drop('age_group', axis=1, inplace=True)
                    self.cleaning_report.append(f"✓ Filled {afc_missing} AFC values using median by age - rounded to integers")
        
        # STRATEGY 2: E2_day5 - Median by protocol, keep 2 decimals
        if 'E2_day5' in self.df.columns:
            e2_missing = self.df['E2_day5'].isnull().sum()
            if e2_missing > 0:
                self.df['E2_day5'] = self.df.groupby('Protocol')['E2_day5'].transform(
                    lambda x: x.fillna(x.median())
                )
                # Round to 2 decimal places for E2 values
                self.df['E2_day5'] = self.df['E2_day5'].round(2)
                self.cleaning_report.append(f"✓ Filled {e2_missing} E2_day5 values using median by protocol")
        
        # STRATEGY 3: n_Follicles - Must be INTEGER (discrete count)
        if 'n_Follicles' in self.df.columns:
            follicles_missing = self.df['n_Follicles'].isnull().sum()
            if follicles_missing > 0:
                # Use simple linear relationship: n_Follicles correlates with AFC and AMH
                mask_complete = self.df[['AFC', 'AMH', 'n_Follicles']].notna().all(axis=1)
                mask_missing = self.df['n_Follicles'].isna()
                
                if mask_complete.sum() > 10:  # Need enough complete cases
                    X_train = self.df.loc[mask_complete, ['AFC', 'AMH']]
                    y_train = self.df.loc[mask_complete, 'n_Follicles']
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    X_pred = self.df.loc[mask_missing, ['AFC', 'AMH']]
                    predictions = model.predict(X_pred)
                    predictions = np.maximum(0, predictions)  # Ensure non-negative
                    predictions = np.round(predictions).astype(int)  # MUST BE INTEGER
                    
                    self.df.loc[mask_missing, 'n_Follicles'] = predictions
                    
                    self.cleaning_report.append(
                        f"✓ Predicted {follicles_missing} n_Follicles values using AFC+AMH regression - rounded to integers"
                    )
                else:
                    # Fallback to median
                    self.df['n_Follicles'].fillna(self.df['n_Follicles'].median(), inplace=True)
                    self.cleaning_report.append(f"✓ Filled {follicles_missing} n_Follicles using median")
            
            # Ensure n_Follicles is integer
            self.df['n_Follicles'] = self.df['n_Follicles'].round().astype(int)
        
        # STRATEGY 4: AMH - Keep 2 decimal places (standard lab precision)
        if 'AMH' in self.df.columns:
            amh_missing = self.df['AMH'].isnull().sum()
            if amh_missing > 0:
                self.df['age_group'] = pd.cut(
                    self.df['Age'], 
                    bins=[0, 30, 35, 50], 
                    labels=['<30', '30-35', '>35']
                )
                self.df['AMH'] = self.df.groupby('age_group')['AMH'].transform(
                    lambda x: x.fillna(x.median())
                )
                self.df.drop('age_group', axis=1, inplace=True)
                self.cleaning_report.append(f"✓ Filled {amh_missing} AMH values using median by age")
            
            # Round AMH to 2 decimal places (standard lab precision)
            self.df['AMH'] = self.df['AMH'].round(2)
        
        # STRATEGY 5: Age - should be integer
        if 'Age' in self.df.columns:
            age_missing = self.df['Age'].isnull().sum()
            if age_missing > 0:
                self.df['Age'].fillna(self.df['Age'].median(), inplace=True)
                self.cleaning_report.append(f"✓ Filled {age_missing} Age values using median")
            self.df['Age'] = self.df['Age'].round().astype(int)
        
        # STRATEGY 6: cycle_number - should be integer
        if 'cycle_number' in self.df.columns:
            cycle_missing = self.df['cycle_number'].isnull().sum()
            if cycle_missing > 0:
                self.df['cycle_number'].fillna(self.df['cycle_number'].mode()[0], inplace=True)
                self.cleaning_report.append(f"✓ Filled {cycle_missing} cycle_number values using mode")
            self.df['cycle_number'] = self.df['cycle_number'].round().astype(int)
        
        # Check if any missing values remain
        missing_after = self.df.isnull().sum()
        if missing_after.sum() > 0:
            print("\n⚠ Remaining missing values after imputation:")
            for col, count in missing_after[missing_after > 0].items():
                print(f"  {col}: {count}")
                # Fill any remaining with overall median
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print("\n✓ Missing value imputation complete")
        print(f"Total missing values after: {self.df.isnull().sum().sum()}")
    
    def standardize_response_labels(self):
        """Ensure response labels are consistent"""
        
        print("\nStandardizing response labels...")
        
        # Convert to lowercase and strip whitespace
        self.df['Patient Response'] = self.df['Patient Response'].str.lower().str.strip()
        
        # Validate labels
        valid_labels = ['low', 'optimal', 'high']
        invalid = ~self.df['Patient Response'].isin(valid_labels)
        
        if invalid.any():
            self.cleaning_report.append(
                f"⚠ Found {invalid.sum()} invalid response labels"
            )
        else:
            self.cleaning_report.append(
                f"✓ All response labels valid: {valid_labels}"
            )
    
    def validate_data_ranges(self):
        """Validate that data falls within clinically reasonable ranges"""
        
        print("\nValidating data ranges...")
        
        validations = {
            'Age': (18, 50, "years"),
            'AMH': (0, 20, "ng/mL"),
            'n_Follicles': (0, 50, "follicles"),
            'E2_day5': (0, 5000, "pg/mL"),
            'AFC': (0, 50, "follicles"),
            'cycle_number': (1, 10, "cycles")
        }
        
        for col, (min_val, max_val, unit) in validations.items():
            if col in self.df.columns:
                out_of_range = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
                
                if out_of_range > 0:
                    self.cleaning_report.append(
                        f"⚠ {out_of_range} values in {col} outside expected range [{min_val}-{max_val} {unit}]"
                    )
                else:
                    self.cleaning_report.append(
                        f"✓ All {col} values within valid range"
                    )
    
    def handle_outliers(self, method='iqr'):
        """Detect and handle outliers"""
        
        print("\nChecking for outliers...")
        
        numeric_columns = ['Age', 'AMH', 'n_Follicles', 'E2_day5', 'AFC']
        
        for col in numeric_columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for medical data
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    self.cleaning_report.append(
                        f"⚠ Found {outliers} potential outliers in {col} (not removed - may be clinically valid)"
                    )
    
    def create_derived_features(self):
        """Create additional features that may be useful"""
        
        print("\nCreating derived features...")
        
        # Age groups
        self.df['age_group'] = pd.cut(
            self.df['Age'], 
            bins=[0, 30, 35, 50], 
            labels=['young', 'middle', 'advanced']
        )
        
        # AMH categories (based on clinical thresholds)
        self.df['amh_category'] = pd.cut(
            self.df['AMH'],
            bins=[0, 1.0, 3.5, float('inf')],
            labels=['low', 'normal', 'high']
        )
        
        # Follicle to AFC ratio (ovarian response efficiency)
        self.df['follicle_afc_ratio'] = self.df['n_Follicles'] / (self.df['AFC'] + 1)
        self.df['follicle_afc_ratio'] = self.df['follicle_afc_ratio'].round(2)
        
        self.cleaning_report.append(
            "✓ Created derived features: age_group, amh_category, follicle_afc_ratio"
        )
    
    def get_summary_statistics(self):
        """Generate summary statistics"""
        
        summary = {
            'total_patients': len(self.df),
            'features': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'response_distribution': self.df['Patient Response'].value_counts().to_dict(),
            'protocol_distribution': self.df['Protocol'].value_counts().to_dict(),
        }
        
        return summary
    
    def clean(self):
        """Execute full cleaning pipeline"""
        
        print("\n" + "="*60)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*60)
        print(f"Original shape: {self.original_shape}")
        print()
        
        # Execute cleaning steps
        patient_mapping = self.de_identify_patients()
        self.standardize_protocols()
        self.standardize_response_labels()
        self.handle_missing_values()
        self.validate_data_ranges()
        self.handle_outliers()
        self.create_derived_features()
        
        # Generate summary
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("CLEANING REPORT:")
        print("="*60)
        for item in self.cleaning_report:
            print(item)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS:")
        print("="*60)
        print(f"Total patients: {summary['total_patients']}")
        print(f"Total features: {summary['features']}")
        print(f"\nResponse distribution:")
        for response, count in summary['response_distribution'].items():
            print(f"  {response}: {count}")
        print(f"\nProtocol distribution:")
        for protocol, count in summary['protocol_distribution'].items():
            print(f"  {protocol}: {count}")
        
        # Show data types
        print(f"\n Data Types:")
        print(f"  AFC: {self.df['AFC'].dtype} (should be int)")
        print(f"  n_Follicles: {self.df['n_Follicles'].dtype} (should be int)")
        print(f"  Age: {self.df['Age'].dtype} (should be int)")
        print(f"  AMH: {self.df['AMH'].dtype} (should be float)")
        
        print("="*60 + "\n")
        
        return self.df, summary, patient_mapping


def main():
    """Main execution"""
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "processed" / "patient_data_with_pdf.csv"
    output_path = project_root / "data" / "processed" / "cleaned_data.csv"
    
    # If PDF data doesn't exist yet, use raw data
    if not input_path.exists():
        input_path = project_root / "data" / "raw" / "patients.csv"
        print(f"⚠ Using raw data from: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Clean data
    cleaner = DataCleaner(df)
    cleaned_df, summary, mapping = cleaner.clean()
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"✓ Saved cleaned data to: {output_path}")
    
    # Save mapping (for audit purposes)
    if mapping:
        mapping_path = project_root / "data" / "processed" / "patient_mapping.csv"
        pd.DataFrame(list(mapping.items()), columns=['original', 'deidentified']).to_csv(
            mapping_path, index=False
        )
        print(f"✓ Saved patient mapping to: {mapping_path}")
    
    return cleaned_df


if __name__ == "__main__":
    main()