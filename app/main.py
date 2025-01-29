# app/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import os

# Define the input data schema using Pydantic
class InputData(BaseModel):
    MedInc: float
    AveRooms: float
    AveOccup: float

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Load the model during startup
model_path = os.path.join("model", "linear_regression_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Home route to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Prediction route to handle form submission
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    MedInc: float = Form(...),
    AveRooms: float = Form(...),
    AveOccup: float = Form(...)
):
    # Prepare the data for prediction
    input_features = [[MedInc, AveRooms, AveOccup]]
    
    # Make prediction using the loaded model
    prediction = model.predict(input_features)
    
    # Return the prediction result with the form
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "prediction": f"Predicted House Price: {prediction[0]:.2f}",
            "MedInc": MedInc,
            "AveRooms": AveRooms,
            "AveOccup": AveOccup
        }
    )