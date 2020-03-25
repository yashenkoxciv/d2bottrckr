import os
import cv2
import yaml
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import misc
from detector import Detector
from tracker import Tracker


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--reid_threshold', default=0.65, type=float)
parser.add_argument('--only_n_frames', default=np.inf, type=float)
args = parser.parse_args()

# load config
with open(args.config, 'r') as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
cfg = argparse.Namespace(**y)

# load detector
detector = Detector(
    cfg.detector_config, cfg.detector_weights,
    reid_weights=cfg.reid_weights,
    roi_threshold=cfg.detector_roi_threshold
)

# instantiate tracker

tracker = Tracker(
    cfg.distance_threshold, cfg.depthmask_file,
    cfg.scenemask_file, cfg.scenemask_threshold,
    cfg.frame_threshold
)

# open video file
iv = cv2.VideoCapture(cfg.video)
total_frames = int(iv.get(cv2.CAP_PROP_FRAME_COUNT))

# open output file
os.makedirs(cfg.output_dir, exist_ok=True)
ov_filename = os.path.join(cfg.output_dir, os.path.split(cfg.video)[1])
ov_codec = cv2.VideoWriter_fourcc(*cfg.codec)
ov = cv2.VideoWriter(ov_filename, ov_codec, cfg.output_fps, (cfg.output_w, cfg.output_h))

for n_frame in tqdm(range(total_frames)):
    if n_frame >= args.only_n_frames:
        break
    
    # read next frame
    ret, frame = iv.read()

    detections = detector(frame)
    #tracker.filter_detections([detections])

    tracker.update(detections)

    misc.draw_detections(frame, detections)
    misc.draw_targets(frame, tracker)

    # write next frame
    ov.write(frame)
    #import ipdb; ipdb.set_trace()

iv.release()
ov.release()
#import ipdb; ipdb.set_trace()

""" m_t, m_l = [], []
for target_idx in tracker.targets:
    target = tracker.targets[target_idx]
    m_t.append(len(target.track))

for target_idx in tracker.lost_targets:
    target = tracker.lost_targets[target_idx]
    m_l.append(len(target.track))

print(np.mean(m_t), np.std(m_t), np.min(m_t), np.max(m_t))
print(np.mean(m_l), np.std(m_l), np.min(m_l), np.max(m_l))
 """