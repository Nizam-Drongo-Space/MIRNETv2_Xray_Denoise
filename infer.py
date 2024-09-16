import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from glob import glob
import cv2
from tqdm import tqdm
import  os
from mirnetv2 import *

weights = './best_model.pth'
input_dir = './noised'
out_dir = './de-noised'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(out_dir, exist_ok=True)

parameters = {
    'inp_channels':3,
    'out_channels':3,
    'n_feat':80,
    'chan_factor':2.0,
    'n_RRG':4,
    'n_MRB':2,
    'height':3,
    'width':2,
    'bias':False,
    'scale':3,
    'task': None
    }
    
    
model = MIRNet_v2(**parameters)
checkpoint = torch.load(weights, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
if device == 'cuda':
    model.cuda()

files = (glob(os.path.join(input_dir, '*')))
img_multiple_of = 4

with torch.no_grad():
  for filepath in tqdm(files):
      img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
      original_height, original_width = img.shape[:2]
      max_size = 512 ## change according to memory consumption

      # Calculate the aspect ratio
      aspect_ratio = original_width / original_height
      if aspect_ratio > 1:
          new_width = max_size
          new_height = int(max_size / aspect_ratio)
      else:
          new_height = max_size
          new_width = int(max_size * aspect_ratio)

      # Resize the image with the new dimensions
      resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
      
      input_ = torch.from_numpy(resized_img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

      # Pad the input if not_multiple_of 4
      h,w = input_.shape[2], input_.shape[3]
      H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
      padh = H-h if h%img_multiple_of!=0 else 0
      padw = W-w if w%img_multiple_of!=0 else 0
      input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

      restored = model(input_)
      restored = torch.clamp(restored, 0, 1)

      # Unpad the output
      restored = restored[:,:,:h,:w]

      restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
      restored = img_as_ubyte(restored[0])

      filename = os.path.split(filepath)[-1]
      cv2.imwrite(os.path.join(out_dir, filename),cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))