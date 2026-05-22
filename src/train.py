import pandas as pd
import mlflow
import mlflow.xgboost
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

# =====================================================
# CONFIG
# =====================================================

RUN_NAME = "exp_4"

FEATURES = [
    "ley_mot_alim_ro",
    "ley_cut_alim_ro",
    "ley_fe_alim_ro",
    "re_mo_total",
    "ley_cus_alim_ro"
]

TARGET = "ley_conc_mo_total_4ta_limpieza"

PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "random_state": 42
}

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/data_model.csv")

df = df[FEATURES + [TARGET]].copy()

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df = df.dropna()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_experiment("Mo_Concentrate_XGBoost")

with mlflow.start_run(run_name=RUN_NAME):

    model = XGBRegressor(**PARAMS)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(
        y_test,
        pred
    ) ** 0.5

    mlflow.log_params(PARAMS)

    mlflow.log_metrics(
        {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }
    )

    # ==========================================
    # REGISTRAR MODELO EN MLFLOW
    # ==========================================

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model"
    )

    # ==========================================
    # GUARDAR MODELO LOCAL
    # ==========================================

    joblib.dump(
        model,
        f"models/{RUN_NAME}.pkl"
    )

    print("=" * 50)
    print(f"Run Name : {RUN_NAME}")
    print(f"R2       : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print("=" * 50)