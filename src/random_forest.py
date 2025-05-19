"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""
import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from mlflow.models import infer_signature
import mlflow


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: balanced DataFrame
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

    # Save and log the transformer
    joblib.dump(col_transf, "column_transformer.pkl")
    mlflow.log_artifact("column_transformer.pkl", artifact_path="transformer")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForestClassifier: trained random forest model
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Infer model signature
    signature = infer_signature(X_train, rf_model.predict(X_train))

    # Log the model with signature
    mlflow.sklearn.log_model(rf_model, "model", signature=signature)

    # Return the trained model
    return rf_model


def main(max_iter):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bank-consumer-churn-prediction-random-forest")

    # Resolve absolute file path
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    file_path = os.path.normpath(os.path.join(script_dir, "../data/Churn_Modelling.csv"))

    with mlflow.start_run():
        df = pd.read_csv(file_path)

        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        # Log max_iter param (even if unused for RF)
        mlflow.log_param("max_iter", max_iter)

        # Train and log model
        model = train(X_train, y_train)

        # Log dataset file as artifact
        mlflow.log_artifact(file_path, artifact_path="data")

        y_pred = model.predict(X_test)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))

        mlflow.set_tag("model", "random_forest")

        # Plot confusion matrix and save BEFORE logging
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix.png")

        mlflow.log_artifact("confusion_matrix.png", artifact_path="confusion_matrix")

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Model")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for the model (unused for RF, logged for consistency)",
    )
    args = parser.parse_args()
    main(args.max_iter)
