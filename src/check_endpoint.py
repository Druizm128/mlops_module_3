from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def request_get_message():
    r = client.get("https://my-ml-app-dante.herokuapp.com/")
    print(r.json())


def request_person_less_than_or_equal_50k():
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
    r = client.post("https://my-ml-app-dante.herokuapp.com/predict_income/", json=data)
    print(r.json())


def request_person_morethan_50k():
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
    print(r.json())



if __name__ == '__main__':
    request_get_message()
    request_person_less_than_or_equal_50k()
    request_person_morethan_50k()

