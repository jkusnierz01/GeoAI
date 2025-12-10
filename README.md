# GeoAI

1. Setup environment (uv)
```bash
uv venv --python=3.11
source .venv/bin/activate
uv sync   # installs from pyproject.toml/uv.lock
```
To reproduce exact versions use the generated `uv.lock`.

2. Get data
    - GEOJSONS - prepare raw geojsons:
        1. `python src/scripts/data_scripts/download_data.py`
        2. `python src/scripts/data_scripts/hexes_to_graph.py -i amenities_hexagons_res7.geojson -o graph_res7.pt`
        3. `python src/scripts/data_scripts/load_graph.py -i graph_res7.pt`
    - GRAPHDATA - preprocessed graph dataset (downloadable):
        - Drive link: https://drive.google.com/drive/folders/1CSBtGa-myD6OSVbABom1LA6rffnNaa3h?usp=drive_link

3. Training
    - Hydra configs live in `src/configs/` — make sure `paths` and other groups point to correct folders before running.
    - Start training (uses Lightning + Hydra):
        ```bash
        # runs with hydra config `src/configs/train.yaml`
        python src/train.py
        ```
    - If you use Weights & Biases, run `wandb login` first and enable the logger in configs.

4. Useful scripts (in `src/scripts`)
    - `src/scripts/data_scripts` : data preparation and preprocessing (see `preprocess_graphs.py`)
    - `src/scripts/evaluate_graph_embeddings` : evaluation helpers (eigenvalues, scale-similarity)
    - `src/scripts/visualize` : embedding visualization helpers (t-SNE, map plots)

5. Notebooks
    - Notebook experiments are placed under `src/notebooks/`.

7. Hydra configs
    - The code uses Hydra for configuration. The main config is `src/configs/train.yaml` and it composes groups from `src/configs/`.
    - Before running, check `src/configs/paths/default.yaml` and update `data_dir` / `output_dir` if needed.

8. Tips
    - If you run into GPU/device or logger errors, check `src/configs/trainer/default.yaml` and `src/configs/logger/wandb.yaml` first.
    - Preprocess your dataset with `src/scripts/data_scripts/preprocess_graphs.py` and use `--scale minmax` or `--scale standard` to compute and save scalers.

9. Quick commands
```bash
# preprocess graphs and save aligned dataset
python src/scripts/data_scripts/preprocess_graphs.py -i dataset -o src/data/dataset_aligned --scale minmax --save_scaler

# train (default hydra config)
python src/train.py

# visualize embeddings (example)
python src/scripts/visualize/visualize_embeddings.py --dataset src/data/dataset_aligned --model_path outputs/checkpoint.pt --city_name berlin --mode map
```