# GeoAI

1. python3 download_data.py
2. python3 hexes_to_graph.py -i amenities_hexagons_res7.geojson -o graph_res7.pt
3. python3 load_graph.py -i graph_res7.pt

Setup GraphMAE:
git checkout pyg
export PYTHONPATH=$(pwd)/GraphMAE:$PYTHONPATH

4. python3 train_unsupervised.py     --dataset custom     --device 0     --encoder gat     --decoder gat     --mask_rate 0.75     --max_epoch 100     --lr 0.001     --save_model
5. python3 show_embeddings.py     --graph
_path graph_res7.pt     --model_path checkpoint.pt     --num_samples 500 