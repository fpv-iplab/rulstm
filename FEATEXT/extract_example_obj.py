import torch
import numpy as np
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image
import lmdb
from tqdm import tqdm
from os.path import basename

env = lmdb.open('features/obj', map_size=1099511627776)
video_name = 'P01_01_frame_{:010d}.jpg'
detections = np.load('data/sample_obj.npy', allow_pickle=True, encoding='bytes')

for i, dets in enumerate(tqdm(detections,'Extracting features')):
    feat = np.zeros(352, dtype='float32')
    for d in dets:
        feat[int(d[0])]+=d[5]
    key = video_name.format(i+1)
    with env.begin(write=True) as txn:
        txn.put(key.encode(),feat)


