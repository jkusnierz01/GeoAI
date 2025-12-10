import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_log_error

benchmark = "san_francisco_crime"
model_type = "LinearRegression"

if benchmark == "airbnb":
    from benchmarks.airbnb import name, prep_dataset, cats, drop, dataset_loader
elif benchmark == "king_county":
    from benchmarks.king_county import name, prep_dataset, cats, drop, dataset_loader
elif benchmark == "san_francisco_crime":
    from benchmarks.san_francisco_crime import name, prep_dataset, cats, drop, dataset_loader
elif benchmark == "chicago_crime":
    from benchmarks.chicago_crime import name, prep_dataset, cats, drop, dataset_loader
elif benchmark == "philadelphia_crime":
    from benchmarks.philadelphia_crime import name, prep_dataset, cats, drop, dataset_loader
else:
    raise ValueError(f"Unknown benchmark: {benchmark}")

if model_type == "RandomForestRegressor":
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
elif model_type == "XGBRegressor":
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
elif model_type == "LinearRegression":
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
else:
    raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(model, dataset_loader, preprocess_fn, categorical_cols, cols_to_drop):
    """
    Generic function to load, process, train, and evaluate any SRAI regression dataset.
    """
    print(f"\n{'='*10} Running Experiment {'='*10}")
    
    data = dataset_loader.load()
    target = dataset_loader.target
    train_gdf = data["train"].copy()
    test_gdf = data["test"].copy()
    
    for df in [train_gdf, test_gdf]:
        df['lat'] = df.geometry.y
        df['lon'] = df.geometry.x
        
    train_gdf = preprocess_fn(train_gdf)
    test_gdf = preprocess_fn(test_gdf)
    
    all_data = pd.concat([train_gdf, test_gdf], keys=['train', 'test'])
    all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    X_train_full = all_data_encoded.loc['train']
    X_test_full = all_data_encoded.loc['test']
    
    drop_final = cols_to_drop + [target, 'geometry']
    features = [c for c in X_train_full.columns if c not in drop_final]
    
    X_train = X_train_full[features].fillna(0)
    X_test = X_test_full[features].fillna(0)
    
    y_train = train_gdf[target]
    y_test = test_gdf[target]
    
    print(f"Features: {len(features)}")
    
    reg = model
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    y_pred_safe = np.maximum(y_pred, 0)
    y_test_safe = np.maximum(y_test, 0)
    
    print(f"--- Results ---")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE:      {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    try:
        print(f"RMSLE:    {root_mean_squared_log_error(y_test_safe, y_pred_safe):.4f}")
    except ValueError:
        print("RMSLE:    Skipped (Target contains negative values?)")

    if model_type in ["RandomForestRegressor", "XGBRegressor"]:
        importances = pd.Series(reg.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))

    return reg


print(f"Running {name} Experiment...")
run_experiment(
    model=model,
    dataset_loader=dataset_loader,
    preprocess_fn=prep_dataset,
    categorical_cols=cats,
    cols_to_drop=drop
)
