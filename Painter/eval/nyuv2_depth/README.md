To perform depth map inference on a single file, use `infer_single_depth.py`.

If more prompts are needed from the NYU V2 data, download 
[nyu_depth_v2_labeled.mat](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
and place it under `data/nyuv2_depth`. Finally, use `load_depth_data.py` in that directory to load
the data into `data/nyuv2_depth/loaded_data`.
