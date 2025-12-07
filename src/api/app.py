from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List
import sys
from pathlib import Path
import logging

# Add parent directory to path to import predict module
sys.path.append(str(Path(__file__).parent.parent / "model"))
from predict import PatientPredictor

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="IVF Patient Response Prediction API",
    description="API for predicting IVF patient response (low/optimal/high)",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (load model once at startup)
predictor = PatientPredictor()


# Request model
class PatientData(BaseModel):
    """Patient data for prediction"""
    Age: int = Field(..., ge=18, le=50, description="Patient age in years")
    AMH: float = Field(..., ge=0, le=20, description="Anti-Müllerian Hormone level (ng/mL)")
    n_Follicles: int = Field(..., ge=0, le=50, description="Number of follicles retrieved")
    E2_day5: float = Field(..., ge=0, le=5000, description="Estradiol level on day 5 (pg/mL)")
    AFC: int = Field(..., ge=0, le=50, description="Antral Follicle Count")
    cycle_number: int = Field(..., ge=1, le=10, description="IVF cycle attempt number")
    Protocol: str = Field(..., description="Stimulation protocol (fixed antagonist, flexible antagonist, or agonist)")

    class Config:
        schema_extra = {
            "example": {
                "Age": 32,
                "AMH": 2.5,
                "n_Follicles": 12,
                "E2_day5": 450.0,
                "AFC": 15,
                "cycle_number": 1,
                "Protocol": "flexible antagonist"
            }
        }


# Response model
class PredictionResponse(BaseModel):
    """Prediction response for a single patient"""
    prediction: str = Field(..., description="Predicted response category (low/optimal/high)")
    confidence: float = Field(..., description="Confidence of the prediction (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")


# Batch prediction response
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "message": "API is running and model is loaded"
    }


@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": type(predictor.model).__name__,
        "classes": list(predictor.label_encoder.classes_),
        "features": predictor.feature_names,
        "feature_count": len(predictor.feature_names)
        # Optionally: add dynamic metrics from training if available
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Make a prediction for a single patient
    """
    try:
        patient_dict = patient.dict()
        valid_protocols = ['fixed antagonist', 'flexible antagonist', 'agonist']

        if patient_dict['Protocol'] not in valid_protocols:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid protocol. Must be one of: {valid_protocols}"
            )

        logging.info(f"Predicting patient: {patient_dict}")
        result = predictor.predict(patient_dict)

        return PredictionResponse(**result)

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(patients: List[PatientData]):
    """
    Make predictions for multiple patients
    Returns a list of PredictionResponse objects for consistency
    """
    try:
        results = []
        for idx, patient in enumerate(patients):
            patient_dict = patient.dict()
            logging.info(f"Predicting patient {idx}: {patient_dict}")
            pred = predictor.predict(patient_dict)
            results.append(PredictionResponse(**pred))

        return {"predictions": results, "count": len(results)}

    except Exception as e:
        logging.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Removed root "/" endpoint and "/docs" references since FastAPI provides /docs automatically

# Run with: 
# cd src/api
# uvicorn app:app --reload 
# Access at: http://127.0.0.1:8000/docs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
