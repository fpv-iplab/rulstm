import torch
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image
import lmdb
from tqdm import tqdm
from os.path import basename
from argparse import ArgumentParser

env = lmdb.open('features/rgb', map_size=1099511627776)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = bninception(pretrained=None)
state_dict = torch.load('models/TSN-rgb.pth.tar')['state_dict']
state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)


model.last_linear = nn.Identity()
model.global_pool = nn.AdaptiveAvgPool2d(1)

model.to(device)

transform = transforms.Compose([
    transforms.Resize([256, 454]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[[2,1,0],...]*255), #to BGR
    transforms.Normalize(mean=[104, 117, 128],
                         std=[1, 1, 1]),
])

imgs = sorted(glob('data/sample_rgb/*.jpg'))

model.eval()
for im in tqdm(imgs,'Extracting features'):
    key = basename(im)
    img = Image.open(im)
    data = transform(img).unsqueeze(0).to(device)
    feat = model(data).squeeze().detach().cpu().numpy()
    with env.begin(write=True) as txn:
        txn.put(key.encode(),feat.tobytes())


