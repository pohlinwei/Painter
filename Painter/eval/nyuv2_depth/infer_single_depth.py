# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import models_painter


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    # build model
    model = getattr(models_painter, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.eval()
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)
    
    loss, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 10000, 0, 10000)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.mean(-1).int()
    output = Image.fromarray(output.numpy())
    output.save(out_path)
    

def get_args_parser():
    parser = argparse.ArgumentParser('NYU Depth V2', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--test_img_path', type=str, 
                        help='path to image whose depth map needs to be computed', default='data/test_images/cat_small.jpg')
    parser.add_argument('--output_path_dir', type=str, 
                        help='path for the output of test_img\'s depth map', default='eval/nyuv2_depth/inferred_depth_maps')
    parser.add_argument('--prompt_img_path', type=str, 
                        help='path to image whose depth map is known', default='data/nyuv2_depth/loaded_data/image_0.png')
    parser.add_argument('--prompt_depth_map_path', type=str, 
                        help='path to depth map that corresponds to prompt_img\'s', default='data/nyuv2_depth/loaded_data/rawDepth_0.npy')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='study_room_0005b/rgb_00094')
    return parser.parse_args()

def _process_image(image_path: str, height: int, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    img = Image.open(image_path).convert("RGB")
    size = img.size
    img = img.resize((height, width))
    img = np.array(img) / 255.
    return img, size

def _process_depth_map(depth_map_path: str, height: int, width: int) -> np.ndarray:
    depth_map = np.load(depth_map_path)
    depth_map = np.array(depth_map) / 10000.
    depth_map = depth_map * 255
    depth_map = Image.fromarray(depth_map).convert("RGB")
    depth_map = depth_map.resize((height, width))
    depth_map = np.array(depth_map) / 255.
    return depth_map

if __name__ == '__main__':
    args = get_args_parser()

    # -- Prompt/Test Image and Depth Map Setup --
    test_img_path = args.test_img_path
    prompt_img_path = args.prompt_img_path
    prompt_depth_map_path = args.prompt_depth_map_path
    
    # --- Model and Device Setup ---
    ckpt_path = args.ckpt_path
    model_painter = prepare_model(ckpt_path, args.model)
    print('Model loaded.')

    # Change to "cpu" if you don't have a CUDA-enabled GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_painter.to(device)

    # --- Single Image Processing Block ---
    res, hres = args.input_size, args.input_size
    # 1. Load the images and concatenate them to create the input image.
    print(f"Loading test image: {test_img_path}")
    test_img, size = _process_image(test_img_path, height=hres, width=res)
    print(f"Loading test image: {prompt_img_path}")
    prompt_img, _ = _process_image(prompt_img_path, height=hres, width=res)    
    # Concatenate prompt and test images.
    input_img = np.concatenate((prompt_img, test_img), axis=0)
    assert input_img.shape == (2 * res, res, 3)
    # Normalize
    input_img -= imagenet_mean
    input_img /= imagenet_std
    # Get test image size.

    # 3. Load the depth maps, including placeholder, and concatenate them to create input depthmap.
    print(f"Loading target image (placeholder): {prompt_depth_map_path}")
    placeholder_depth_map = _process_depth_map(prompt_depth_map_path, height=hres, width=res)
    prompt_depth_map = _process_depth_map(prompt_depth_map_path, height=hres, width=res)
    # Concatenate placeholder_depth_map and prompt_depth_map.
    input_depth_map = np.concatenate((prompt_depth_map, placeholder_depth_map), axis=0)
    assert input_depth_map.shape == (2 * res, res, 3)
    # Normalize
    input_depth_map -= imagenet_mean
    input_depth_map /= imagenet_std

    # 4. Run inference
    test_file_name = os.path.splitext(os.path.basename(test_img_path))[0]
    output_path = os.path.join(args.output_path_dir, f'{test_file_name}.png')
    print(f"Running inference and saving output to: {output_path}")
    torch.manual_seed(2)
    run_one_image(input_img, input_depth_map, size, model_painter, output_path, device)
    print("Inference complete.")
