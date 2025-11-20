# GeoAI

1. Get data
    * Prepare data:
        1. `python3 download_data.py`
        2. `python3 hexes_to_graph.py -i amenities_hexagons_res7.geojson -o graph_res7.pt`
        3. `python3 load_graph.py -i graph_res7.pt`
    * or download
        * `gdown --folder https://drive.google.com/drive/folders/1ZWoCOlIi2mOZQgQwsLb9Y_QxgDWfZtlU -O dataset --remaining-ok`


2. Setup GraphMAE:
    1. `git checkout pyg`
    2. `export PYTHONPATH=$(pwd)/GraphMAE:$PYTHONPATH`

3. Train: 
    * `python3 train_unsupervised.py --dataset dataset --device 0 --encoder gat --decoder gat --mask_rate 0.75 --max_epoch 100 --lr 0.001 --save_model`
4. Check embeddings 
    * `python3 show_embeddings.py --dataset dataset --model_path checkpoint.pt --samples_per_graph 100`