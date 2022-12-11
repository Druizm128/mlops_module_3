# API code

import os
import joblib
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from ml.model import inference
from ml.data import process_data
from train_model import cat_features

model_path = "../model"
##### Load machine learning artifacts #####
# Loading model and encoders:
model = joblib.load(os.path.join(model_path, "inference_model.pkl"))
encoder = joblib.load(os.path.join(model_path, "onehot_encoder.pkl"))
lb = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
# Loading data
test = pd.read_csv("../data/test.csv")

##### Define data model and validations #####
class Person(BaseModel):
    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

##### Instantiate app #####
app=FastAPI()

# Define a GET on the specified endpoint
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome ... this app will predict your income!"}

# Define a POST that generates an inference
@app.post("/predict_income/")
async def get_income(body: Person):
    # Convert the Person payload to a dictionary
    test = pd.DataFrame(data = [body.dict(by_alias=True)])
    # Preprocess payload
    X_test, _, _, _ = process_data(
        test, 
        categorical_features=cat_features, 
        label=None,
        training=False, 
        encoder=encoder, 
        lb=lb
    )
    # Predict income
    pred_income = inference(model, X_test)
    # Preprocess output
    if pred_income[0]:
        pred = {'salary': '>50k'}
    else:
        pred = {'salary': '<=50k'}
    # Return json with body and prediction
    return {
        "body": body, 
        "income_prediction": pred
    }