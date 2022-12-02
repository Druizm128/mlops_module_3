import json
import pytest
import pandas as pd

# Load train data
@pytest.fixture(scope="session")
def data_train():
    df=pd.read_csv("data/train.csv")
    return df

# Load test data
@pytest.fixture(scope="session")
def data_test():
    df=pd.read_csv("data/test.csv")
    return df

# Load prep train data
@pytest.fixture(scope="session")
def data_x_train():
    df=pd.read_csv("data/X_train.csv")
    return df

# Load prep test data
@pytest.fixture(scope="session")
def data_x_test():
    df=pd.read_csv("data/X_test.csv")
    return df

# Load model results
@pytest.fixture(scope="session")
def model_results():
    with open('logs/model_results.json', 'r') as openfile:
        json_object = json.load(openfile)
    return json_object

# Test pandas data frame with obs
def test_train_length(data_train):
    '''Test that we have enough data in the trainig set'''
    assert len(data_train) > 1000

def test_inference_length(data_test):
    '''Test that we have observations in the inference set'''
    assert len(data_test) > 1000

# Test that the prep data has the same number of columns
def test_equal_columns(data_x_train, data_x_test):
    '''Test that the train and the inference sets have equal number of columns'''
    assert data_x_train.shape[1] == data_x_test.shape[1]

# Test model recall is at least X
def test_minumum_recall(model_results):
    '''Check for a minimum recall level'''
    assert model_results['recall'] > 0.60

# Test model precision is at least X
def test_minimum_precision(model_results):
    '''Check for a minimum precision level'''
    assert model_results['precision'] > 0.60