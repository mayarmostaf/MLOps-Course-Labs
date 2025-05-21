"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction using XGBoost.
"""

import os
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
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

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier


def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    churn_maj, churn_min = (churn_0, churn_1) if len(churn_0) > len(churn_1) else (churn_1, churn_0)
    churn_maj_downsample = resample(churn_maj, n_samples=len(churn_min), replace=False, random_state=1234)
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df, artifacts_path):
    features = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]

    df = df[features]
    df_bal = rebalance(df)

    X = df_bal.drop("Exited", axis=1)
    y = df_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    X_train = pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train,
                           columns=col_transf.get_feature_names_out())
    X_test = pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test,
                          columns=col_transf.get_feature_names_out())

    os.makedirs(artifacts_path, exist_ok=True)
    transformer_path = os.path.join(artifacts_path, "column_transformer.pkl")
    joblib.dump(col_transf, transformer_path)
    mlflow.log_artifact(transformer_path, artifact_path="transformer")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train, max_iter, n_estimators):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=n_estimators
    )
    model.fit(X_train, y_train)

    signature = mlflow.models.infer_signature(X_train, y_train)
    mlflow.xgboost.log_model(model, "model", signature=signature)

    return model


def main(max_iter, n_estimators):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, "../data/Churn_Modelling.csv"))
    artifacts_path = os.path.join(script_dir, "artifacts")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bank-consumer-churn-prediction-xgboost")

    with mlflow.start_run():
        df = pd.read_csv(data_path)
        _, X_train, X_test, y_train, y_test = preprocess(df, artifacts_path)

        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_estimators", n_estimators)

        model = train(X_train, y_train, max_iter, n_estimators)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))

        mlflow.set_tag("model", "xgboost")
        mlflow.log_artifact(data_path, artifact_path="data")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
        disp.plot()
        conf_mat_path = os.path.join(artifacts_path, "confusion_matrix.png")
        plt.savefig(conf_mat_path)
        mlflow.log_artifact(conf_mat_path, artifact_path="confusion_matrix")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and log an XGBoost model with MLflow.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations (used internally).")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosting rounds for XGBoost.")
    args = parser.parse_args()
    main(args.max_iter, args.n_estimators)
