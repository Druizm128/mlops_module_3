# Script to train machine learning model.

# Add the necessary imports for the starter code.
import logging
import json
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
# Add code to load in the data.
logging.basicConfig(
    filename='../logs/model_training.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info("---START---")
    logging.info("Loading data")
    data=pd.read_csv("../data/census_clean.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logging.info("Splitting train and test data")
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    logging.info("Preprocessing training data")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    logging.info("Preprocessing testing data")
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # Train and save a model.
    logging.info("Training model")
    best_model=train_model(X_train, y_train)
    logging.info("Predicting on test data")
    y_pred=inference(best_model, X_test)
    logging.info("Computing metrics")
    precision, recall, fbeta=compute_model_metrics(y_test, y_pred)
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"fbeta: {fbeta}")

    logging.info("Saving artifacts")
    logging.info("Saving datasets")
    train.to_csv("../data/train.csv", index=False)
    test.to_csv("../data/test.csv", index=False)
    pd.DataFrame(X_train).to_csv("../data/X_train.csv", index=False, header=None)
    pd.DataFrame(X_test).to_csv("../data/X_test.csv", index=False, header=None)

    logging.info("Saving one hot encoder")
    with open("../model/onehot_encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)

    logging.info("Saving label encoder")
    with open("../model/label_encoder.pkl", "wb") as file:
        pickle.dump(lb, file)

    logging.info("Saving inference model")
    with open("../model/inference_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    
    logging.info("Saving model results dictionary")
    json_object=json.dumps({'precision': precision, 'recall': recall})
    with open("../logs/model_results.json", "w") as outfile:
        outfile.write(json_object)

    logging.info("---FINISHED---")