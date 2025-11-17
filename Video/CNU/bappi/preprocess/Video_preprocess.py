import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import freqs
from tqdm import tqdm
import os
import math

input_file = './data/Symptoms_Parkinson’s_dise.mp4'
output_dir = './output_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    print('error:...')
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count/fps

    print(f" Video Info:")
    print(f" Resolution: {width}x{height}")
    print(f" FPS: {fps}")
    print(f" Total Frames: {frame_count}")
    print(f" Duration: {duration:.2f} seconds")

# Extract first N frames for visualization
start_frame = 290
N = 30
frames = []
print(f"frames: {N}")
frame_ids = [start_frame + i for i in range(N)]

for fid in frame_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

# Visualize frames
cols = 8
rows = math.ceil(N/cols)
plt.figure(figsize=(10, rows * 2))
for idx, frame in enumerate(frames):
    plt.subplot(rows, cols, idx+1)
    plt.imshow(frame)
    plt.title(f"Frame {frame_ids[idx]}")
    plt.axis('off')

plt.suptitle("Sample Video Frames")
plt.tight_layout()
plt.show()