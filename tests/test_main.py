# Tests for the API


from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_get_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "greeting": "Welcome ... this app will predict your income!"}


def test_less_than_or_equal_50k():
    data = {
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
    r = client.post("/predict_income/", json=data)
    assert r.status_code == 200
    print(r.json())
    assert r.json()['income_prediction'] == {"salary": "<=50k"}


def test_morethan_50k():
    data = {
        "age": 51,
        "workclass": "Private",
        "fnlgt": 110747,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": " White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1887,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/predict_income/", json=data)
    assert r.status_code == 200
    assert r.json()['income_prediction'] == {"salary": ">50k"}
