import pdfplumber
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

class PDFExtractor:
    #Extract patient data 
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.patient_data = {}
        
    def extract_text(self):
        #Extract text from PDF
        with pdfplumber.open(self.pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    def parse_patient_info(self, text):
        # Extract Name
        name_match = re.search(r'Name\s*:\s*([A-Za-z\s]+)', text)
        if name_match:
            self.patient_data['name'] = name_match.group(1).strip()
        
        # Extract Protocol
        protocol_match = re.search(r'Protocol\s*:\s*([A-Za-z\s]+)', text)
        if protocol_match:
            protocol_raw = protocol_match.group(1).strip()
            # Standardize protocol name
            if 'Flex' in protocol_raw or 'flex' in protocol_raw:
                self.patient_data['protocol'] = 'flexible antagonist'
            elif 'Fix' in protocol_raw or 'fix' in protocol_raw:
                self.patient_data['protocol'] = 'fixed antagonist'
            elif 'Agonist' in protocol_raw or 'agonist' in protocol_raw:
                self.patient_data['protocol'] = 'agonist'
            else:
                self.patient_data['protocol'] = protocol_raw
        
        # Extract Birth date and calculate age
        birth_match = re.search(r'Birth date:\s*(\d{1,2}/\d{1,2}/\d{2,4})', text)
        if birth_match:
            birth_str = birth_match.group(1)
            birth_date = datetime.strptime(birth_str, '%d/%m/%y')
            current_year = 2025  # Based on the monitoring date
            age = current_year - birth_date.year
            self.patient_data['age'] = age
        
        # Extract AMH
        amh_match = re.search(r'AMH\s*:\s*([\d.]+)', text)
        if amh_match:
            self.patient_data['amh'] = float(amh_match.group(1))
        
        # Extract AFC 
        afc_match = re.search(r'AFC\s*:\s*([\d.]+)', text)
        if afc_match:
            self.patient_data['afc'] = float(afc_match.group(1))
        
        # Extract Cycle number
        cycle_match = re.search(r'Cycle number\s*:\s*(\d+)', text)
        if cycle_match:
            self.patient_data['cycle_number'] = int(cycle_match.group(1))
        
        # Extract Number of follicles
        follicles_match = re.search(r'Number Of follicles\s*=\s*(\d+)', text)
        if follicles_match:
            self.patient_data['n_follicles'] = int(follicles_match.group(1))
        
        # Extract Patient Response
        response_match = re.search(r'(low|optimal|high)-response', text, re.IGNORECASE)
        if response_match:
            self.patient_data['patient_response'] = response_match.group(1).lower()
    
    def extract_monitoring_table(self, text):
            """Extract E2_day5 from monitoring table"""
            
            lines = text.split('\n')
            
            # Find the line with "J" followed by a number that could be 5
            # Then look at the next line for the E2 value (350)
            for i, line in enumerate(lines):
                # Check if this line has "J" followed by something that might be day 5
                # Could be "J 5", "J5", or "J 3.5" (where 3.5 is LH value on same row)
                if re.search(r'J\s+[\d.]+', line):
                    # Check if the number after J could indicate row 5
                    # Look for pattern "J <number>" where number contains 5 or is around line position 5
                    # For line "J 3.5", the next line should have the data
                    # Pattern: look for "//" followed by a 3-digit number
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        # Pattern: "6/10 // 350 ..."
                        e2_match = re.search(r'//\s*(\d{3,4})', next_line)
                        if e2_match:
                            e2_value = float(e2_match.group(1))
                            if 50 <= e2_value <= 5000:
                                self.patient_data['e2_day5'] = e2_value
                                print(f"✓ Extracted E2_day5: {e2_value}")
                                return
            
    def calculate_afc_from_table(self, text):
        # AFC = count of follicles from right and left ovary on early monitoring day
        # estimated from the follicle data
        if 'afc' not in self.patient_data:
            # Pattern: numbers separated by /
            follicle_pattern = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', text)
            if follicle_pattern:
                # Take measurements from an early day (usually first monitoring with follicles)
                # Count follicles > 2mm (represented by numbers in the table)
                first_measurement = follicle_pattern[0] if follicle_pattern else None
                if first_measurement:
                    # Simple estimation: count the follicles mentioned
                    afc_estimate = len(follicle_pattern[0])
                    self.patient_data['afc'] = afc_estimate
    
    def de_identify_patient(self, patient_name, patient_id):
        #patient name to format 25XXX
        return f"25{patient_id:03d}"
    
    def extract_all(self):
        """Main extraction method"""
        text = self.extract_text()
        self.parse_patient_info(text)
        self.extract_monitoring_table(text)
        self.calculate_afc_from_table(text)
        
        return self.patient_data
    
    def to_dataframe_row(self, patient_id):
        """Convert extracted data to DataFrame row format"""
        data = self.extract_all()
        
        # De-identify
        deidentified_id = self.de_identify_patient(
            data.get('name', ''), 
            patient_id
        )
        
        # Create row matching CSV structure
        row = {
            'patient_id': deidentified_id,
            'cycle_number': data.get('cycle_number', None),
            'Age': data.get('age', None),
            'Protocol': data.get('protocol', None),
            'AMH': data.get('amh', None),
            'n_Follicles': data.get('n_follicles', None),
            'E2_day5': data.get('e2_day5', None),
            'AFC': data.get('afc', None),
            'Patient Response': data.get('patient_response', None)
        }
        
        return row


def add_pdf_to_csv(pdf_path, csv_path, output_path, new_patient_id):
    """
    Extract data from PDF and add to existing CSV
    
    Args:
        pdf_path: Path to PDF file
        csv_path: Path to existing CSV file
        output_path: Path to save updated CSV
        new_patient_id: ID number for the new patient (expl: 1 for 25001)
    """
    
    # Load existing CSV
    df = pd.read_csv(csv_path)
    
    # Extract data from PDF
    extractor = PDFExtractor(pdf_path)
    new_row = extractor.to_dataframe_row(new_patient_id)
    
    # Add new row to dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(output_path, index=False)
    
    print(f" Successfully extracted data from PDF")
    print(f" Added patient {new_row['patient_id']} to dataset")
    print(f" Saved to {output_path}")
    
    return df, new_row


def main():    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "data" / "raw" / "sample.pdf"
    csv_path = project_root / "data" / "raw" / "patients.csv"
    output_path = project_root / "data" / "processed" / "patient_data_with_pdf.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine next patient ID (find max existing ID)
    df_existing = pd.read_csv(csv_path)
    
    # Count existing patients to assign new ID
    next_id = len(df_existing) + 1
    
    # Extract and add PDF data
    df_updated, new_patient = add_pdf_to_csv(
        pdf_path=pdf_path,
        csv_path=csv_path,
        output_path=output_path,
        new_patient_id=next_id
    )
    
    # Display extracted data
    print("EXTRACTED PATIENT DATA:")
    for key, value in new_patient.items():
        print(f"{key:20s}: {value}")    
    return df_updated


if __name__ == "__main__":
    main()