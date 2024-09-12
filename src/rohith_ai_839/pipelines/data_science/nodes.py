import logging
from typing import Dict, Tuple
import plotly.io as pio
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from evidently import ColumnMapping
import plotly.graph_objects as go
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
import json
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *
from os.path import abspath


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LogisticRegression(penalty="l2", C=0.01)
    regressor.fit(X_train, y_train)
    return regressor

def quality_drift_check(X_train: pd.DataFrame, X_test: pd.DataFrame, ):
    report = Report(metrics=[
    DataDriftPreset(), 
])
    report.run(reference_data=X_train, current_data=X_test)
    return json.loads(report.json())


def evaluate_model(
    regressor: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
)->pd.Series:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", score)
    
    return y_pred


def prediction_drift_check(y_test: pd.Series, y_pred: pd.Series):
    report = Report(metrics=[
    DataDriftPreset(), 
])
    report.run(reference_data= y_test.to_frame(name="y"), current_data=pd.DataFrame(y_pred, columns = ["y"]))
    report.save_html("data/08_reporting/evidently_plot.html")
    return json.loads(report.json())

def plot_and_save(column_name, current_data, reference_data):
    current_x = current_data['x']
    current_y = current_data['y']
    
    reference_x = reference_data['x']
    reference_y = reference_data['y']
    
    # Create the plot
    fig = go.Figure()

    # Add trace for current data
    fig.add_trace(go.Scatter(x=current_x, y=current_y, mode='lines+markers', name='Current Distribution'))

    # Add trace for reference data
    fig.add_trace(go.Scatter(x=reference_x, y=reference_y, mode='lines+markers', name='Reference Distribution'))

    # Update layout
    fig.update_layout(title=f'Distributions for {column_name}',
                      xaxis_title='X values',
                      yaxis_title='Probability Density',
                      legend=dict(x=0.02, y=0.98),
                      template='plotly_dark')

    # Show the plot
    pio.write_image(fig, file="data/08_reporting/{}_distribution.png".format(column_name.replace("/","_")))


def report_plotly(data_drift, pred_drift):
    print(type(data_drift))
    data_drift_by_columns = data_drift['metrics'][1]['result']['drift_by_columns']
    pred_drift_by_columns = pred_drift['metrics'][1]['result']['drift_by_columns']
    for column, data in data_drift_by_columns.items():
        current_distribution = data['current']['small_distribution']
        reference_distribution = data['reference']['small_distribution']
        plot_and_save(column, current_distribution, reference_distribution)
    for column, data in pred_drift_by_columns.items():
        current_distribution = data['current']['small_distribution']
        reference_distribution = data['reference']['small_distribution']
        plot_and_save(column, current_distribution, reference_distribution)
