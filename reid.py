import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
add_size  = [12, 6]
target_size = [256, 128]
normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
transform = T.Compose([T.Resize(target_size),
                       T.ToTensor(),
                       normalize_transform])


class ReIdentification:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = torch.load(checkpoint).eval()
        self.model.classifier = torch.nn.Sequential()
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            normalize_transform
        ])
    
    def __call__(self, image):
        pil_image = Image.fromarray(image)
        im = self.transform(pil_image)[None, :, :, :]
        with torch.no_grad():
            z = self.model(im.cuda()).cpu().numpy()
        z = z / np.linalg.norm(z)
        return z[0]

#reid = ReIdentification('configs/resnet50_model_120.pth')
#import ipdb; ipdb.set_trace()
