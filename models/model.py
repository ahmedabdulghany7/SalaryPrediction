import pickle
import numpy as np
from typing import Dict

class SalaryPredictor:
    def __init__(self):
        # Load models
        with open("rf_model.pkl", "rb") as f:
            self.rf_model = pickle.load(f)
        with open("nn_model.pkl", "rb") as f:
            self.nn_model = pickle.load(f)
        with open("lr_model.pkl", "rb") as f:
            self.lr_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            self.label_encoders = pickle.load(f)

    def preprocess_input(self, input_data: Dict) -> np.ndarray:
        # Encode categorical variables
        encoded_features = []
        encoded_features.append(input_data['years_exp'])
        
        categorical_cols = ['title', 'work_type', 'work_hour', 'city', 'company_type']
        for col in categorical_cols:
            encoded_val = self.label_encoders[col.title()].transform([input_data[col]])[0]
            encoded_features.append(encoded_val)
            
        # Scale features
        return self.scaler.transform(np.array(encoded_features).reshape(1, -1))

    def predict(self, input_data: Dict) -> Dict:
        processed_input = self.preprocess_input(input_data)
        
        rf_pred = self.rf_model.predict(processed_input)[0]
        nn_pred = self.nn_model.predict(processed_input)[0]
        lr_pred = self.lr_model.predict(processed_input)[0]
        
        # Weighted ensemble prediction
        ensemble_pred = 0.5 * nn_pred + 0.3 * rf_pred + 0.2 * lr_pred
        
        return {
            "random_forest_prediction": round(float(rf_pred), 2),
            "neural_network_prediction": round(float(nn_pred), 2),
            "linear_regression_prediction": round(float(lr_pred), 2),
            "ensemble_prediction": round(float(ensemble_pred), 2)
        }