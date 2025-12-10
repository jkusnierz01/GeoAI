import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_log_error
import rootutils

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.file_utils import load_and_merge_embeddings

def run_experiment(model, dataset_loader, preprocess_fn, categorical_cols, cols_to_drop, use_dino, dino_path, use_graph, graph_path, target_col=None, output_file=None):
    # Use dataset_loader.target if target_col is not provided
    target = target_col if target_col else dataset_loader.target
    
    print(f"\n{'='*10} Running Experiment {'='*10}")
    
    data = dataset_loader.load()
    train_gdf = data["train"].copy()
    test_gdf = data["test"].copy()
    
    # 1. Base Preprocessing
    for df in [train_gdf, test_gdf]:
        if 'geometry' in df.columns:
            df['lat'] = df.geometry.y
            df['lon'] = df.geometry.x
        
    train_gdf = preprocess_fn(train_gdf)
    test_gdf = preprocess_fn(test_gdf)
    
    # 2. MERGE EMBEDDINGS
    if use_dino:
        print(f"Merging DINO embeddings from {dino_path}")
        train_gdf = load_and_merge_embeddings(train_gdf, dino_path, "dino")
        test_gdf = load_and_merge_embeddings(test_gdf, dino_path, "dino")
        
    if use_graph:
        print(f"Merging GraphMAE embeddings from {graph_path}")
        train_gdf = load_and_merge_embeddings(train_gdf, graph_path, "graph")
        test_gdf = load_and_merge_embeddings(test_gdf, graph_path, "graph")

    # 3. Encoding & Feature Selection
    all_data = pd.concat([train_gdf, test_gdf], keys=['train', 'test'])
    all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    X_train_full = all_data_encoded.loc['train']
    X_test_full = all_data_encoded.loc['test']
    
    drop_final = cols_to_drop + [target, 'geometry']
    # h3_index is usually dropped, but if not in cols_to_drop, remove it too
    if 'h3_index' in X_train_full.columns:
        drop_final.append('h3_index')
        
    features = [c for c in X_train_full.columns if c not in drop_final]
    
    X_train = X_train_full[features].fillna(0)
    X_test = X_test_full[features].fillna(0)
    
    y_train = train_gdf[target]
    y_test = test_gdf[target]
    
    print(f"Total Features: {len(features)}")
    
    # 4. Train
    reg = model
    reg.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = reg.predict(X_test)
    
    y_pred_safe = np.maximum(y_pred, 0)
    y_test_safe = np.maximum(y_test, 0)
    
    print(f"--- Results ---")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.2f}")
    print(f"RMSE:     {rmse:.2f}")
    
    rmsle = None
    try:
        rmsle = root_mean_squared_log_error(y_test_safe, y_pred_safe)
        print(f"RMSLE:    {rmsle:.4f}")
    except ValueError:
        print("RMSLE:    Skipped (Target contains negative values?)")

    # Feature Importance (Only for trees)
    top_features = {}
    if hasattr(reg, "feature_importances_"):
        importances = pd.Series(reg.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))
        top_features = importances.head(10).to_dict()

    if output_file:
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "use_graph": use_graph,
            "use_dino": use_dino,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "rmsle": rmsle,
            "top_features": top_features
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Append to list if file exists, else create new list
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            except json.JSONDecodeError:
                data = []
        else:
            data = []
            
        data.append(results)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nResults saved to {output_file}")

    return reg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="philadelphia_crime", help="Benchmark name")
    parser.add_argument("--model_type", type=str, default="XGBRegressor", help="Model type")
    parser.add_argument("--use_dino", action="store_true", help="Use DINO embeddings")
    parser.add_argument("--dino_path", type=str, default="embeddings/dino_v3_embeddings.pkl")
    parser.add_argument("--use_graph", action="store_true", help="Use GraphMAE embeddings")
    parser.add_argument("--graph_path", type=str, default="embeddings/graphmae_embeddings.pkl")
    parser.add_argument("--output_file", type=str, default="src/outputs/benchmarks/results.json", help="Path to save results JSON")
    
    args = parser.parse_args()

    if args.benchmark == "airbnb":
        from src.benchmarks.airbnb import name, prep_dataset, cats, drop, dataset_loader
    elif args.benchmark == "king_county":
        from src.benchmarks.king_county import name, prep_dataset, cats, drop, dataset_loader
    elif args.benchmark == "san_francisco_crime":
        from src.benchmarks.san_francisco_crime import name, prep_dataset, cats, drop, dataset_loader
    elif args.benchmark == "chicago_crime":
        from src.benchmarks.chicago_crime import name, prep_dataset, cats, drop, dataset_loader
    elif args.benchmark == "philadelphia_crime":
        from src.benchmarks.philadelphia import name, prep_dataset, cats, drop, dataset_loader
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    if args.model_type == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif args.model_type == "XGBRegressor":
        from xgboost import XGBRegressor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use 'hist' tree method for GPU support in modern XGBoost
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, device=device, tree_method="hist" if device == "cuda" else "auto")
    elif args.model_type == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif args.model_type == "DeepRegressor":
        from src.benchmarks.mlp import DeepRegressor
        model = DeepRegressor(epochs=50, lr=0.001, batch_size=32)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    run_experiment(
        model=model,
        dataset_loader=dataset_loader,
        preprocess_fn=prep_dataset,
        categorical_cols=cats,
        cols_to_drop=drop,
        use_dino=args.use_dino,
        dino_path=args.dino_path,
        use_graph=args.use_graph,
        graph_path=args.graph_path,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
