# Script that test the model performance in categorical levels.

import pickle
import os

import pandas as pd

from src.ml.model import compute_model_metrics, inference
from src.ml.data import process_data
from src.train_model import cat_features


def check_slices_performance(data, model_path):
    '''
    Estimates model metric in slices of data based on the categorical values
    of the variables

    Params:
    ------
    data: DataFrame
    model_path: str

    Return:
    ------
    DataFrame with the performance results
    '''

    # Loading model and encoders:
    with open(os.path.join(model_path, "inference_model.pkl"), "rb") as file:
        model = pickle.load(file)

    with open(os.path.join(model_path, "onehot_encoder.pkl"), "rb") as file:
        encoder = pickle.load(file)

    with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as file:
        lb = pickle.load(file)

    # Processing data:
    X_test, y_test, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    # Measuring model performance:
    predictions = inference(model, X_test)

    # Calculating slice results:
    slice_results = []
    for group in cat_features:
        performance = (
            test
            .assign(y_true=y_test, y_pred=predictions)
            .groupby([group])
            .apply(
                lambda df: compute_model_metrics(df["y_true"], df["y_pred"]))
            .apply(pd.Series)
            .rename(columns={
                0: 'precision',
                1: 'recall',
                2: 'fbeta'})
            .round(2)
            .reset_index()
            .assign(category=group)
            .rename(columns={group: 'subcategory'})
            .filter([
                'category',
                'subcategory',
                'precision',
                'recall',
                'fbeta']))
        slice_results.append(performance)
    return pd.concat(slice_results)


if __name__ == "__main__":

    # Loading the data:
    test = pd.read_csv("data/test.csv")
    model_path = "model"
    # Calculating slice performance:
    output = check_slices_performance(test, model_path)

    # Printing results to file:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_rows', None)
    print(output)

    # Save results as a log
    output.to_csv("logs/slice_output.csv", index=False)
