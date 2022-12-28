'''
This script tests the API to return the status code and
income prediction using a sample person.
'''
import requests
import json

person_1 = {
        "age": "57",
        "workclass": "Private",
        "fnlgt": 153918,
        "education": "HS-grad",
        "education-num": "9",
        "marital-status": "Married-civ-spouse",
        "occupation": "Transport-moving",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": "0",
        "capital-loss": "0",
        "hours-per-week": "40",
        "native-country": "United-States"
    }

print("Sending a Post method to get income prediction")
r = requests.post('https://my-ml-app-dante.herokuapp.com/predict_income/', json=person_1)
print(f"Status code: {r.status_code}")
print(f"Income prediction: {json.loads(r.content)['income_prediction']}")