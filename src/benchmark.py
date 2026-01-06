import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_log_error
from sklearn.decomposition import PCA
import rootutils

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.file_utils import load_and_merge_embeddings
from src.benchmarks.mlp import DeepRegressor

def run_experiment(model, dataset_loader, preprocess_fn, categorical_cols, cols_to_drop, use_dino, dino_path, dino_pca, use_graph, graph_path, graph_pca, target_col=None, output_file=None, only_embeddings=False, benchmark_name=None):
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
        
    if only_embeddings:
        features = []
        if use_dino:
            features += [c for c in X_train_full.columns if c.startswith("dino_") or c.startswith("dino_pca_")]
        if use_graph:
            features += [c for c in X_train_full.columns if c.startswith("graph_") or c.startswith("graph_pca_")]
        if not features:
            print("Warning: No embedding features selected. Please enable --use_dino and/or --use_graph.")
    else:
        features = [c for c in X_train_full.columns if c not in drop_final]

    X_train = X_train_full[features].fillna(0)
    X_test = X_test_full[features].fillna(0)

    if isinstance(model, (DeepRegressor,)): # Add other NN models here if needed
        print("Applying StandardScaler for DeepRegressor...")
        scaler = StandardScaler()
            
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Apply PCA if specified
    dino_features = [col for col in X_train.columns if col.startswith("dino_")]
    if use_dino and dino_features and dino_pca > 0:
        pca = PCA(n_components=min(dino_pca, len(dino_features)))
        X_train_dino = X_train[dino_features]
        X_test_dino = X_test[dino_features]

        X_train_pca = pca.fit_transform(X_train_dino)
        X_test_pca = pca.transform(X_test_dino)

        for df, pca_data, name in zip([X_train, X_test], [X_train_pca, X_test_pca], ["train", "test"]):
            for col in dino_features:
                df.drop(col, axis=1, inplace=True)
            for i in range(X_train_pca.shape[1]):
                df[f"dino_pca_{i+1}"] = pca_data[:, i]

    graph_features = [col for col in X_train.columns if col.startswith("graph_")]
    if use_graph and graph_features and graph_pca > 0:
        pca = PCA(n_components=min(graph_pca, len(graph_features)))
        X_train_graph = X_train[graph_features]
        X_test_graph = X_test[graph_features]

        X_train_pca = pca.fit_transform(X_train_graph)
        X_test_pca = pca.transform(X_test_graph)

        for df, pca_data, name in zip([X_train, X_test], [X_train_pca, X_test_pca], ["train", "test"]):
            for col in graph_features:
                df.drop(col, axis=1, inplace=True)
            for i in range(X_train_pca.shape[1]):
                df[f"graph_pca_{i+1}"] = pca_data[:, i]
    
    y_train = train_gdf[target]
    y_test = test_gdf[target]

    features = X_train.columns.tolist()
    
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

    if output_file:
        results = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "use_graph": use_graph,
            "use_dino": use_dino,
            "use_only_embeddings": only_embeddings,
            "num_features": len(features),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "rmsle": rmsle
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
    parser.add_argument("--dino_pca", type=int, default=0, help="Number of PCA components for DINO embeddings, if 0, no PCA applied")
    parser.add_argument("--use_graph", action="store_true", help="Use GraphMAE embeddings")
    parser.add_argument("--graph_path", type=str, default="embeddings/graphmae_embeddings.pkl")
    parser.add_argument("--graph_pca", type=int, default=0, help="Number of PCA components for GraphMAE embeddings, if 0, no PCA applied")
    parser.add_argument("--output_file", type=str, default="src/outputs/benchmarks/results.json", help="Path to save results JSON")
    parser.add_argument("--only_embeddings", action="store_true", help="Use only graph/dino embeddings, ignore benchmark features")
    
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
    elif args.benchmark == "beijing_housing":
        from src.benchmarks.beijing_housing import name, prep_dataset, cats, drop, dataset_loader
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
        model = DeepRegressor(epochs=100, lr=0.0005, batch_size=512)
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
        dino_pca=args.dino_pca,
        use_graph=args.use_graph,
        graph_path=args.graph_path,
        graph_pca=args.graph_pca,
        output_file=args.output_file,
        only_embeddings=args.only_embeddings,
        benchmark_name=args.benchmark
    )

if __name__ == "__main__":
    main()
