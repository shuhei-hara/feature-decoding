from glob import glob
import os

from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.models import AlexNet
from bdpy.util import makedir_ifnot
import numpy as np
import PIL
from scipy.io import savemat
import torch
import re

from torchvision import transforms
import torchvision
from torchvision.models.alexnet import AlexNet_Weights

# Setting ####################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_dir = ''
# image_ext = 'jpg'

output_dir = ''

img_size = (224, 224)  # (width, height); this should match to the input size of the encoder DNN
mean_image = np.float32([104., 117., 123.])

encoder_param_file = '<path to model parameters pt file>'

layers = [
    'conv1', 'conv2', 'conv3', 'conv4',
    'conv5', 'fc6', 'fc7', 'fc8'
]

layer_mapping = {
    'conv1': 'features[0]',
    'conv2': 'features[3]',
    'conv3': 'features[6]',
    'conv4': 'features[8]',
    'conv5': 'features[10]',
    'fc6':   'classifier[1]',
    'fc7':   'classifier[4]',
    'fc8':   'classifier[6]',
}


transform = transforms.Compose([
    transforms.Resize(img_size,interpolation=transforms.InterpolationMode.BICUBIC),  # already set to (224, 224)
    transforms.ToTensor(),  # converts (H, W, C) in [0,255] to (C, H, W) in [0.0–1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet RGB mean
                         std=[0.229, 0.224, 0.225])   # ImageNet RGB std
])



# Main #######################################################################


encoder = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT).to(device)
encoder.eval()

feature_extractor = FeatureExtractor(encoder, layers, layer_mapping, device=device, detach=True)

image_files = []
for ext in ['jpg', 'JPEG']:
    image_files.extend(glob(os.path.join(image_dir, '*.' + ext)))
image_files = sorted(image_files)
print(image_files)

for image_file in image_files:
    print(image_file)

    img = PIL.Image.open(image_file).convert('RGB')
    img = transform(img)  # img is now a torch.Tensor of shape (3, 224, 224)
    img = img.unsqueeze(0).to(device) 

    features = feature_extractor.run(img)

    # Save features
    for layer in features.keys():
        f = features[layer]
        
        # Extract the base filename without extension
        base = os.path.splitext(os.path.basename(image_file))[0]  # e.g., '0100_b30'
        
        # Use regex to extract the image number before '_b'
        # match = re.match(r'^0*([0-9]+)_b\d+$', base)
        match = re.match(r'^0*([0-9]+)$', base)
        if match:
            image_number = match.group(1)  # This removes leading zeros
            output_name = f'{image_number}.mat'
        else:
            output_name = f'{base}.mat'  # fallback (in case pattern doesn't match)
        
        output_file = os.path.join(
            output_dir,
            layer,
            output_name
        )

        makedir_ifnot(os.path.join(output_dir, layer))

        savemat(output_file, {'feat': f})
        print('Saved {}'.format(output_file))
    

print('All done')