from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    title: str = Field(..., description="Job title")
    years_exp: float = Field(..., description="Years of experience")
    work_type: str = Field(..., description="Type of work")
    work_hour: str = Field(..., description="Work hours")
    city: str = Field(..., description="City of company site")
    company_type: str = Field(..., description="Type of company")

class PredictionOutput(BaseModel):
    random_forest_prediction: float
    neural_network_prediction: float
    linear_regression_prediction: float
    ensemble_prediction: float