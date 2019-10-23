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

env = lmdb.open('features/flow', map_size=1099511627776)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = bninception(pretrained=None)
model.conv1_7x7_s2 = nn.Conv2d(10, 64,kernel_size=(7,7), stride=(2,2), padding=(3,3))
state_dict = torch.load('models/TSN-flow.pth.tar')['state_dict']
state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

model.last_linear = nn.Identity()
model.global_pool = nn.AdaptiveAvgPool2d(1)

model.to(device)

transform = transforms.Compose([
    transforms.Resize([256, 454]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255),
    transforms.Normalize(mean=[128],
                         std=[1]),
])

imgs = sorted(glob('data/sample_flow/*_u_*.jpg'))

flow_buffer = []

model.eval()
for im in tqdm(imgs,'Extracting features'):
    key = basename(im).replace('flow_u_','frame_')
    img_u = Image.open(im).convert('L')
    img_v = Image.open(im.replace('_u_','_v_')).convert('L')
    #repeat the first five frames
    for _ in range(1 if len(flow_buffer)>0 else 5):
        flow_buffer.append(transform(img_u))
        flow_buffer.append(transform(img_v))
    if len(flow_buffer)>10:
        del flow_buffer[0]
        del flow_buffer[0]
    if len(flow_buffer)==10:
        data = torch.cat(flow_buffer[-10:],0).unsqueeze(0).to(device)
        feat = model(data).squeeze().detach().cpu().numpy()
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat.tobytes())


