"""
Methods to load data, analyze customer churn,
train models and plot training results
"""

import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

sns.set()
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO)


def map_dtypes(categorical=[], string=[], numeric=[]):
    """Create dtype mapper for pd.read_csv to parse columns as specified dtypes

    Args:
        categorical, string, numeric (list): Column names to parse as type

    Usage:
        >>> dtype_mappevs codr = map_dtypes(categorical_columns=['gender', 'mood'])
        >>> df = pd.read_csv(csv_file.csv, dtype=dtype_mapper)
    """

    dtype_categorical = dict(zip(categorical, ["category" for i in range(len(categorical))]))
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

        # add feature
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        return df

    except FileNotFoundError:
        logging.error(f"Could not find CSV file in {filepath}")


def plot_histograms(df, column_list):
    """Show histograms for specified columns"""

    f, axes = plt.subplots(len(column_list), 1, figsize=(12,6*len(column_list)))
    f.suptitle("Distributions", fontsize=18)

    for row, column in enumerate(column_list):
        try:
            sns.histplot(df[column], ax=axes[row])

        # older seaborn versions do not have histplot
        # use distplot instead
        except AttributeError:
            logging.warning("Using distplot instead of histplot. Consider updating seaborn")
            sns.distplot(df[column], ax=axes[row])


def plot_relative_count(df, column_list):
    """Relative count for specified columns in ascending order"""

    f, axes = plt.subplots(len(column_list), 1, figsize=(12, 6*len(column_list)))
    f.suptitle("Relative counts", fontsize=18)

    for plot_row, column in enumerate(column_list):
        relative_count = (df[column]
                          .value_counts(normalize=True)
                          .mul(100)
                          .reset_index()
                          .rename(columns={'index': column, column: '%'}))

        # with only 1 column, 'axes' is not a list and axes[plot_row] would return TypeError
        if len(column_list) == 1:
            sns.barplot(x=column, y="%", data=relative_count, order=relative_count[column], ax=axes)
        elif len(column_list) > 1:
            sns.barplot(x=column, y="%", data=relative_count, order=relative_count[column], ax=axes[plot_row])


def plot_correlation_heatmap(df):
    """Return correlation heatmap"""
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Relative counts", fontsize=18)
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2, square=True, annot_kws={"size": 14})


def perform_eda(df, describe=False, filepath=None, export_as='pdf'):
    '''
    perform eda on df and save figures to filepath as pdf
    input:
            df: pandas dataframe
            describe (bool): whether to print df.describe()
            filepath (str): directory to store plots
            export_as (str): filetype of saved plots in filepath

    output:
            None
    '''
    print(f"Dimension: {df.shape}")
    print(f"Missings:\n--------------\n{df.isnull().sum()}")
    if describe: print(f"{df.describe()}")

    plot_histograms(df=df, column_list=['Customer_Age', "Total_Trans_Ct"])
    if filepath: plt.savefig(os.path.join(filepath, f'histograms.{export_as}'))

    plot_relative_count(df=df, column_list=['Churn', 'Marital_Status'])
    if filepath: plt.savefig(os.path.join(filepath,f'relative_counts.{export_as}'))

    plot_correlation_heatmap(df=df)
    if filepath: plt.savefig(os.path.join(filepath, f'correlation_heatmap.{export_as}'))


def encoder_helper(df, column_mapper, target_feature):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            column_mapper (dict): dict of columns that contain categorical features and names for created features
            target_feature (str): Feature on which to calculate mean
    output:
            df: pandas dataframe with new columns for
    '''
    if not isinstance(column_mapper, dict): raise TypeError("column_mapper has the be a dict")

    for column, new_feature in column_mapper.items():
        df[new_feature] = df.groupby(column)[target_feature].transform("mean")

    return df


def perform_feature_engineering(df, target_variable, ignore_features=[], test_size=0.3, random_state=42):
    '''Assign target variable (y), select features (x) and return train-test split
    input:
              df: pandas dataframe
              target_variable (str): specify column name of the target variable (y)
              ignore_features (list): which features should be ignored that appear in df
              random_state (int): Random nr. seed for train test split
              test_size (float): Fraction of test set

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing datat

    raises:
        ValueError: When target_variable not among df.columns
        TypeError: Features to ignore must be a list
    '''
    if target_variable not in df.columns:
        raise ValueError(f"Target variable {target_variable} not among df columns.")
    if not isinstance(ignore_features, list):
        raise TypeError(f"ignore_features must be of list type. It is of type {type(ignore_features)}")

    X = df[df.columns.difference(ignore_features+[target_variable])]
    y = df.loc[:,target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state)

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              y_train: y training data
    output:
              tuple: trained random forest model, linear regresssion model
    '''
    # random forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4,5,100],
        'criterion': ['gini', 'entropy']
    }

    logging.info("Start CV for random forest")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # logistic regressor
    lrc = LogisticRegression(max_iter=300)
    logging.info("Train linear classifier")
    lrc.fit(X_train, y_train)

    return cv_rfc, lrc


def classification_report_image(X_train,
                                y_train,
                                X_test,
                                y_test,
                                rf_model,
                                lr_model,
                                filepath,
                                export_as='pdf'):
    '''
    produces classification report for train and test, stores report as image
    in images folder
    input:
            X_train: training features
            y_train: training response values
            X_test: test features
            y_test:  test response values
            rf_model: trained random forest model
            lr_model: trained linear classifier model
            filepath: path and filename (without file extension)
            export_as: file type to export classification_report
    output:
             None
    '''
    # random forest
    y_train_preds_rf = rf_model.best_estimator_.predict(X_train)
    y_test_preds_rf = rf_model.best_estimator_.predict(X_test)

    # linear classifier
    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    preds = {
            'rf': {
                    'test': y_test_preds_rf,
                    'train': y_train_preds_rf
                },
            'lr': {
                    'test': y_test_preds_lr,
                    'train': y_train_preds_lr
                },
            }

    # create classification reports
    for model, pred in preds.items():
        clf_report_test = classification_report(
            y_test,
            pred['test'],
            output_dict=True)
        clf_report_train = classification_report(
            y_train,
            pred['train'],
            output_dict=True)

        # create subplots
        f, axes = plt.subplots(2, 1, figsize=(12, 6*2))
        f.suptitle(f"Classification report: {model.upper()}", fontsize=18)

        axes[0].set_title("Test")
        sns.heatmap(
            pd.DataFrame(clf_report_test).iloc[:-1, :].T,
            annot=True, ax=axes[0])

        axes[1].set_title("Train")
        sns.heatmap(
            pd.DataFrame(clf_report_train).iloc[:-1, :].T,
            annot=True,
            ax=axes[1])
        plt.savefig(
            os.path.join(
                filepath,
                f'{model}_classification_report.{export_as}'))


def feature_importance_plot(model, X_train, filepath='./images', export_as='pdf'):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_train: pandas dataframe of X training features ()
            filepath: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=90)

    plt.savefig(os.path.join(filepath, f'feature_importance.{export_as}'))

