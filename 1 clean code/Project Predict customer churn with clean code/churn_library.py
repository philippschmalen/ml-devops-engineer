"""
Methods to load data, analyze customer churn,
train models and plot training results
"""

import pandas as pd
import logging


def map_dtypes(categorical=[], string=[], numeric=[]):
    """Create dtype mapper for pd.read_csv to parse columns as specified dtypes

    Args:
        categorical, string, numeric (list): Column names to parse as type

    Usage:
        >>> dtype_mapper = map_dtypes(categorical_columns=['gender', 'mood'])
        >>> df = pd.read_csv(csv_file.csv, dtype=dtype_mapper)
    """

    dtype_categorical = dict(
        zip(categorical, ["category" for i in range(len(categorical))])
    )
    dtype_numeric = dict(zip(numeric, ["float" for i in range(len(numeric))]))
    dtype_str = dict(zip(string, ["str" for i in range(len(string))]))

    dtype_mapper = {**dtype_categorical, **dtype_numeric, **dtype_str}

    return dtype_mapper


def import_data(
    filepath, index_col=0, categorical_col=[], numeric_col=[], string_col=[]
):
    """
    returns dataframe for the csv found at pth

    input:
            pth (str): directory that holds the csv
            index_col (int): use specified column as index
    output:
            df: pandas dataframe
    """
    dtype_mapper = map_dtypes(
        categorical=categorical_col, numeric=numeric_col, string=string_col
    )

    try:
        df = pd.read_csv(filepath, index_col=index_col, dtype=dtype_mapper)
        return df

    except FileNotFoundError:
        logging.error(f"Could not find CSV file in {filepath}")


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    print(f"Dimension: {df.shape}")
    print(f"Missings: {df.isnull().sum().to_markdown()}")
    print(f"df.describe()")


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    pass


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass
