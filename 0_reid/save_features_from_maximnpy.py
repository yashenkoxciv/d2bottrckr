import pandas as pd
import utils.datasets as datasets
from utils.utils import plot_one_box
import cv2
import numpy as np
import torchvision.transforms as T
import torch
from PIL import Image
import os
import time
import pickle
import argparse
import pandas as pd
import sys


import argparse

parser = argparse.ArgumentParser(description='Save features')
parser.add_argument('tracks_file', help='npy file with tracks from maxim')
parser.add_argument('video',  help='video file')
parser.add_argument('output',  help='output')

args = parser.parse_args()

# our best reid model (at least for market)
args_code = "/home/seger/work/workspace/AI/reid_workspace/bag_of_tricks_tests/08_IN_test/reid-strong-baseline_INwithaffine"
args_checkpoint = "/home/seger/work/workspace/AI/reid_workspace/bag_of_tricks_tests/08_IN_test/08_msmt17_duke_cuhk03_syri_sysu_viper_ward_merge/Experiment-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_model_120.pth"
args_layer = "bottleneck"

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def xywh2xyxy(x):
    y = np.zeros(4, dtype = np.int)
    y[0] = x[0]
    y[1] = x[1]
    y[2] = x[0] + x[2]
    y[3] = x[1] + x[3]
    return y

def clip_to_size_xyxy(bbx_xyxy, size):
    out = np.zeros(4, dtype = np.int)
    out[0] = np.clip(bbx_xyxy[0], 0, size[0])
    out[1] = np.clip(bbx_xyxy[1], 0, size[1])
    out[2] = np.clip(bbx_xyxy[2], 0, size[0])
    out[3] = np.clip(bbx_xyxy[3], 0, size[1])
    if (not np.allclose(out, bbx_xyxy)):
        print("CLIP!", out, bbx_xyxy)
    return out

sys.path.append(args_code)

size = (1920, 1080)
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
add_size  = [12, 6]
target_size = [256, 128]
normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
transform = T.Compose([T.Resize(target_size),
                       T.ToTensor(),
                       normalize_transform])

features = []

def hook(module, input_, output):
    global features
    f_batch = output.cpu().detach().numpy()
    features += list(f_batch)
                

model = torch.load(args_checkpoint)
model.eval()
exec("model.%s.register_forward_hook(hook)"%args_layer)

def extract_feature(img_cv):
    global features
    features = []    
    img_pil = Image.fromarray(img_cv)    
    im  = transform(img_pil)[None,:,:,:]
    model(im.cuda())
    return features[0]



tracks = np.load(args.tracks_file)


ftracks = []


dataloader = datasets.LoadVideo(args.video, size)

for frame_id, _, img0 in dataloader:                
    print(frame_id)
    for bbx_id, (x,y,h,w,visibility) in enumerate(tracks[:,frame_id,:]):
        if (x < -10000):
            continue
        bbx_xyhw = np.array([x,y,h,w])
        bbx_xyxy = xywh2xyxy(bbx_xyhw)
        bbx_xyxy_clipped = clip_to_size_xyxy(bbx_xyxy, size)
        img_bbx = img0[bbx_xyxy_clipped[1]: bbx_xyxy_clipped[3], bbx_xyxy_clipped[0]: bbx_xyxy_clipped[2]]
        f = extract_feature(img_bbx)
        f = f / np.linalg.norm(f) 
        ftracks.append([frame_id, bbx_id, bbx_xyxy_clipped, f, 1, visibility])

pickle.dump( ftracks, open( args.output, "wb" ) )
