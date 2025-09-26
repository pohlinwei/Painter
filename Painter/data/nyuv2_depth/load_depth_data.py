import h5py
import numpy as np
from PIL import Image

NYUV2_DEPTH_DATA_FILE_PATH = 'data/nyuv2_depth/nyu_depth_v2_labeled.mat'
INDEX = 1
LOADED_DATA_DIR = 'data/nyuv2_depth/loaded_data'

data_raw_depth = None
data_image = None

with h5py.File(NYUV2_DEPTH_DATA_FILE_PATH) as f:
    print(f.keys())
    data_raw_depth = f['rawDepths'][INDEX]
    data_image = f['images'][INDEX].T

    Image.fromarray(data_image).save(f'{LOADED_DATA_DIR}/image_{INDEX}.png')
    np.save(f'{LOADED_DATA_DIR}/rawDepth_{INDEX}.npy', data_raw_depth)
