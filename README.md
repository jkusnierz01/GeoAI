# GeoAI
1. To download submodules (scale-mae): `git submodule update --init --recursive`

2. Setup env.
```
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Get data
    * GEOJSONS - Prepare data:
        1. `python data_scripts/download_data.py`
        2. `python data_scripts/hexes_to_graph.py -i amenities_hexagons_res7.geojson -o graph_res7.pt`
        3. `python data_scripts/load_graph.py -i graph_res7.pt`
    * GRAPHDATA - you can download
        * `gdown --folder https://drive.google.com/drive/folders/1ZWoCOlIi2mOZQgQwsLb9Y_QxgDWfZtlU -O dataset --remaining-ok`


3. Training: 
    * config: `configs/defaults.yaml`
    * if you want weight&biases plots - in `defatults.yaml` you have to set `wandb` flag True and in terminal log to wandb: `wandb login`
    * to run: `python train_unsupervised.py`


4. Embeddings:
    * T-SNE: `python visualize/tsne_embeddings.py --dataset dataset --model_path checkpoint.pt --samples_per_graph 100`
    * Visualize on map: `visualize/visualize_embeddings.py` - YOU NEED GRAPTH data `.pt` and `.geojson` and `model checkpoint`!!
    * Eigenspectrum:  `python evaluate_graph_embeddings/eigenvalue_spectrum.py  --dataset dataset_aligned --model_path checkpoint.pt`
    * Scale similarity (cosine similarity of same and random regions in different resolutions) 
        - `python evaluate_graph_embeddings/scale_similairty.py --dataset dataset_aligned --model_path checkpoint.pt`