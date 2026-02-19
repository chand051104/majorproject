# Models

Drop trained artifacts here:

- `segmentation_model.pt` for satellite segmentation (TorchScript or serialized torch module)
- `temporal_model.pt` for temporal change detection
- `risk_model.pkl` for trained risk fusion model

If `segmentation_model.pt` is missing or not loadable, satellite engine uses heuristic vision over fetched tiles.
If `risk_model.pkl` is missing, the pipeline uses deterministic weighted fusion fallback.
