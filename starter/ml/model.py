from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Train and save a model.
    rf_model=RandomForestClassifier()
    distributions={
        'n_estimators':[50, 100, 120, 150, 200, 250, 300, 400, 500, 800],
        'max_depth':[5, 8, 10, 12, 15, 20, 25, 30, None],
        'min_samples_split':[2,5,10,15],
        'min_samples_leaf':[1,2,5,10]
    }
    grid_search=RandomizedSearchCV(
        rf_model,
        param_distributions=distributions,
        n_iter=10,
        cv=5
    )
    grid_search.fit(X_train, y_train)
    model=grid_search.best_estimator_
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
