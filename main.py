from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.model import SalaryPredictor
from schemas.input_schema import PredictionInput, PredictionOutput

app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting salaries based on various features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = SalaryPredictor()

@app.get("/")
async def root():
    return {"message": "Welcome to the Salary Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict_salary(input_data: PredictionInput):
    try:
        prediction = model.predict({
            "title": input_data.title,
            "years_exp": input_data.years_exp,
            "work_type": input_data.work_type,
            "work_hour": input_data.work_hour,
            "city": input_data.city,
            "company_type": input_data.company_type
        })
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)