import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import lightgbm --remove
from sklearn.ensemble import RandomForestClassifier
# Split the dataframe into test and train data


# original
# def split_data(data_df):
#     """Split a dataframe into training and validation datasets"""

#     features = data_df.drop(['target', 'id'], axis=1)
#     labels = np.array(data_df['target'])
#     features_train, features_valid, labels_train, labels_valid = \
#         train_test_split(features, labels, test_size=0.2,
#                          random_state=0)

#     train_data = lightgbm.Dataset(features_train, label=labels_train)
#     valid_data = lightgbm.Dataset(
#         features_valid,
#         label=labels_valid,
#         free_raw_data=False)

#     return (train_data, valid_data)

# new
def split_data(data_df):
    """Split a dataframe into training and validation datasets"""

    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2,
                         random_state=0)

    # train_data = lightgbm.Dataset(features_train, label=labels_train)
    # valid_data = lightgbm.Dataset(
    #     features_valid,
    #     label=labels_valid,
    #     free_raw_data=False)

    return (features_train, features_valid, labels_train, labels_valid)
# Train the model, return the model
# original
# def train_model(data, parameters):
#     """Train a model with the given datasets and parameters"""
#     # The object returned by split_data is a tuple.
#     # Access train_data with data[0] and valid_data with data[1]

#     train_data = data[0]
#     valid_data = data[1]

#     model = lightgbm.train(parameters,
#                            train_data,
#                            valid_sets=valid_data,
#                            num_boost_round=500,
#                            early_stopping_rounds=20)

#     return model
# new
def train_model(train_x, train_y):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    # train_data = data[0]
    # valid_data = data[1]

    model = RandomForestClassifier(criterion='gini',
                                 n_estimators=200,
                                 random_state=42,
                                 n_jobs=3)
    model.fit(train_x, train_y)

    return model


# Evaluate the metrics for the model
# original
# def get_model_metrics(model, data):
#     """Construct a dictionary of metrics for the model"""
#     predictions = model.predict(data[1].data)
#     fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
#     model_metrics = {
#         "auc": (
#             metrics.auc(
#                 fpr, tpr))}
#     print(model_metrics)

#     return model_metrics
# new
def get_model_metrics(model,test_x,test_y):
    """Construct a dictionary of metrics for the model"""
    predictions = model.predict(test_x)
    model_metrics = metrics.accuracy_score(test_y,predictions)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, predictions)
    model_metrics = {
        "auc": (
            metrics.auc(
                fpr, tpr))}
    print(model_metrics)

    return model_metrics