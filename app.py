from fastapi import FastAPI, HTTPException
import uvicorn #ASGI
import joblib
import pandas as pd
from pydantic import BaseModel, conint

# Load the trained models
model_top_classifier = joblib.load("top_classifier_model_xtree.pkl")
model_2nd_best_classifier = joblib.load("second_best_classifier_model_lgbm.pkl")
model_ensemble_classifier = joblib.load("ensemble_classifier_model_5.pkl")

# Load the StandardScaler
sc = joblib.load("standard_scaler.pkl")

#defining features where standard scaler will be applied (excluding Soil_type and Wilderness_area_Type
cols_for_scaler = ['Elevation', 'Aspect', 'Slope', 'Vertical_Distance_To_Hydrology',
                  'Horizontal_Distance_To_Roadways', 'Hillshade_Noon', 'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points', 'net_hyd_distance',
                  'mean_distance_horizontal', 'sqrtHorizontal_Distance_To_Hydrology',
                  'Elevation_m_HR']

# Create the FastAPI app
app = FastAPI(title="Forest Cover Type Prediction API")

# Define a data model to receive input from the user
class InputData(BaseModel):
    Elevation: float
    Aspect: float
    Slope: float
    Vertical_Distance_To_Hydrology: float
    Horizontal_Distance_To_Roadways: float
    Hillshade_Noon: conint(ge=0, le=255)
    Hillshade_3pm: conint(ge=0, le=255)
    Horizontal_Distance_To_Fire_Points: float
    net_hyd_distance: float
    mean_distance_horizontal: float
    sqrtHorizontal_Distance_To_Hydrology: float
    Elevation_m_HR: float
    Soil_Type: conint(ge=1, le=40)  # Constrain to values between 1 and 40, inclusive
    Widerness_Area_Type: conint(ge=1, le=4)  # Constrain to values between 1 and 4, inclusive


# Define the index and version endpoints
@app.get("/")
async def index():
    return {"message": "Forest Cover Type Prediction (Rifat)!"}

@app.get("/version")
async def version():
    return {"version": "1.0"}

# Define the prediction endpoints for each model
@app.post("/predict1/")
async def predict_top_classifier(data: InputData):
    # Convert user input to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocess the input data using the StandardScaler
    input_data[cols_for_scaler] = sc.transform(input_data[cols_for_scaler])

    # Make the prediction using the top classifier model
    prediction = model_top_classifier.predict(input_data)

    # Return the prediction
    return {"Cover_Type_Prediction": int(prediction[0])}

@app.post("/predict2/")
async def predict_second_best_classifier(data: InputData):
    # Convert user input to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocess the input data using the StandardScaler
    input_data[cols_for_scaler] = sc.transform(input_data[cols_for_scaler])

    # Make the prediction using the 2nd best classifier model
    prediction = model_2nd_best_classifier.predict(input_data)

    # Return the prediction
    return {"Cover_Type_Prediction": int(prediction[0])}

@app.post("/predict3/")
async def predict_ensemble_classifier(data: InputData):
    # Convert user input to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocess the input data using the StandardScaler
    input_data[cols_for_scaler] = sc.transform(input_data[cols_for_scaler])


    # Make the prediction using the ensemble classifier model
    prediction = model_ensemble_classifier.predict(input_data)

    # Return the prediction
    return {"Cover_Type_Prediction": int(prediction[0])}


if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#run in terminal
# uvicorn filename:object --reload (uvicorn app:app --reload)