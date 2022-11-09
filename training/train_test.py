import lightgbm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

# functions to test are imported from train.py
# from train import split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""

def split_data_gbm(data_df):
    """Split a dataframe into training and validation datasets"""

    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2,
                         random_state=0)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(
        features_valid,
        label=labels_valid,
        free_raw_data=False)
        
    return (train_data, valid_data)

def train_model_gbm(data, parameters):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    train_data = data[0]
    valid_data = data[1]

    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=valid_data,
                           num_boost_round=500,
                           early_stopping_rounds=20)

    return model

def get_model_metrics_gbm(model, data):
    """Construct a dictionary of metrics for the model"""
    predictions = model.predict(data[1].data)
    fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
    model_metrics = {
        "auc": (
            metrics.auc(
                fpr, tpr))}
    print(model_metrics)

    return model_metrics

def test_split_data():
    test_data = {
        'id': [0, 1, 2, 3, 4],
        'target': [0, 0, 1, 0, 1],
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 1, 1, 2, 1]
        }

    data_df = pd.DataFrame(data=test_data)
    # original
    # data = split_data(data_df)
    # new
    data = split_data_gbm(data_df)

    # verify that columns were removed correctly
    assert "target" not in data[0].data.columns
    assert "id" not in data[0].data.columns
    assert "col1" in data[0].data.columns

    # verify that data was split as desired
    assert data[0].data.shape == (4, 2)
    assert data[1].data.shape == (1, 2)


def test_train_model():
    data = __get_test_datasets()

    params = {
        "learning_rate": 0.05,
        "metric": "auc",
        "min_data": 1
    }

    # original
    # model = train_model(data, params)
    # new
    model = train_model_gbm(data, params)

    # verify that parameters are passed in to the model correctly
    for param_name in params.keys():
        assert param_name in model.params
        assert params[param_name] == model.params[param_name]


def test_get_model_metrics():
    class MockModel:

        @staticmethod
        def predict(data):
            return np.array([0, 0])

    data = __get_test_datasets()

    # original
    # metrics = get_model_metrics(MockModel(), data)
    # new
    metrics = get_model_metrics_gbm(MockModel(), data)

    # verify that metrics is a dictionary containing the auc value.
    assert "auc" in metrics
    auc = metrics["auc"]
    np.testing.assert_almost_equal(auc, 0.5)


def __get_test_datasets():
    """This is a helper function to set up some test data"""
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([1, 1, 0, 1, 0, 1])
    X_test = np.array([7, 8]).reshape(-1, 1)
    y_test = np.array([0, 1])

    train_data = lightgbm.Dataset(X_train, y_train)
    valid_data = lightgbm.Dataset(X_test, y_test)
    data = (train_data, valid_data)
    return data
