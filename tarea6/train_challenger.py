import pathlib
import pickle
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from mlflow.models.signature import infer_signature
from prefect import flow, task
from dotenv import load_dotenv
import math

# -------------------------
# Task: Leer datos
# -------------------------
@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df

# -------------------------
# Task: Features
# -------------------------
@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dv = DictVectorizer()

    X_train = dv.fit_transform(df_train[categorical + numerical].to_dict(orient="records"))
    X_val = dv.transform(df_val[categorical + numerical].to_dict(orient="records"))
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

# -------------------------
# Task: Entrenar un modelo
# -------------------------
@task(name="Train Model")
def train_model(X_train, X_val, y_train, y_val, dv, model_suffix: str):
    mlflow.set_registry_uri("databricks-uc")
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "seed": 42
    }

        # Dentro de train_model
    with mlflow.start_run(run_name=f"{model_suffix}_model"):
        booster = xgb.train(params=params, dtrain=train, evals=[(valid, "validation")], num_boost_round=50)
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
    
        # Guardar preprocesador ...
        
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])
    
        # Log model
        mlflow.xgboost.log_model(
            booster,
            artifact_path="model",  # IMPORTANTE: debe coincidir con model_uri
            input_example=input_example,
            signature=signature
        )
    
        # URI exacta para registrar
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        return model_uri, rmse

# -------------------------
# Task: Registrar challenger
# -------------------------
@task(name="Register Challenger")
def register_challenger(model_uri, rmse):
    client = mlflow.MlflowClient()
    model_name = "workspace.default.nyc_taxi_model_prefect"

    # Registrar el modelo como @challenger
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_registered_model_alias(name=model_name, alias="challenger", version=mv.version)
    return mv.version, rmse

# -------------------------
# Task: Comparar y actualizar champion
# -------------------------
@task(name="Compare and Update Champion")
def compare_and_update_champion(challenger_version, challenger_rmse):
    client = mlflow.MlflowClient()
    model_name = "workspace.default.nyc_taxi_model_prefect"

    # Obtener versión actual de @champion
    try:
        champion_version = client.get_registered_model_alias(name=model_name, alias="champion").version
        champion_run = client.get_model_version(name=model_name, version=champion_version)
        champion_rmse = float(champion_run.run_id)  # O usar otra forma de obtener RMSE del run
    except Exception:
        champion_version, champion_rmse = None, float("inf")

    # Comparar RMSE y actualizar alias
    if challenger_rmse < champion_rmse:
        client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
        print(f"✅ Nuevo champion: version {challenger_version}")
    else:
        print(f"✅ Champion permanece: version {champion_version}")

# -------------------------
# Flow principal
# -------------------------
@flow(name="Train Challenger Flow")
def train_challenger_flow(year: int, month_train: str, month_val: str):
    load_dotenv(override=True)
    train_path = f"green_tripdata_{year}-{month_train}.parquet"
    val_path = f"green_tripdata_{year}-{month_val}.parquet"

    df_train = read_data(train_path)
    df_val = read_data(val_path)
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Entrenar dos modelos como ejemplo (puedes variar hyperparams)
    model_uri1, rmse1 = train_model(X_train, X_val, y_train, y_val, dv, "trial_1")
    model_uri2, rmse2 = train_model(X_train, X_val, y_train, y_val, dv, "trial_2")

    # Registrar challenger
    challenger_version, challenger_rmse = register_challenger(model_uri1 if rmse1 < rmse2 else model_uri2,
                                                              min(rmse1, rmse2))

    # Comparar y actualizar @champion
    compare_and_update_champion(challenger_version, challenger_rmse)

if __name__ == "__main__":
    year = 2025
    month_train = "01"
    month_val = "02"

    train_challenger_flow(year=year, month_train=month_train, month_val=month_val)