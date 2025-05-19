"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""
import joblib
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    joblib.dump(col_transf, "column_transformer.pkl")
    mlflow.log_artifact("column_transformer.pkl", artifact_path="transformer")
    return col_transf, X_train, X_test, y_train, y_test

def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)

    # Log model
    signature = mlflow.models.infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(log_reg, "model", signature=signature)

    ### Log the data
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    file_path = os.path.normpath(os.path.join(script_dir, "../data/Churn_Modelling.csv"))
    mlflow.log_artifact(file_path, artifact_path="data")
    return log_reg


def main(max_itr):
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("bank-consumer-churn-prediction-logistic-regression")

    ### Start a new run and leave all the main function code as part of the experiment
    mlflow.start_run()
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    file_path = os.path.normpath(os.path.join(script_dir, "../data/Churn_Modelling.csv"))
    df = pd.read_csv(file_path)
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter
    mlflow.log_param("max_iter", max_itr)
    model = train(X_train, y_train)

    
    y_pred = model.predict(X_test)

    ### Log metrics after calculating them
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))

    ### Log tag
    mlflow.set_tag("model", "logistic_regression")

    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    plt.savefig("confusion_matrix.png")

    # Log the image as an artifact in MLflow
    
    mlflow.log_artifact("confusion_matrix.png", artifact_path="confusion_matrix")
    mlflow.end_run()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression Model")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for the logistic regression model",
    )
    args = parser.parse_args()
    max_iter = args.max_iter
    main(max_iter)
