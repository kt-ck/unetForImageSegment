# import numpy as np
# from skimage.transform import resize
# from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01

# x = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
# y = np.random.randint(10, 15, size=(128, 128), dtype=np.uint8)

# transforms = ComposeDouble([
#     FunctionWrapperDouble(resize,
#                           input=True,
#                           target=False,
#                           output_shape=(128, 128, 3)),
#     FunctionWrapperDouble(resize,
#                           input=False,
#                           target=True,
#                           output_shape=(128, 128),
#                           order=0,
#                           anti_aliasing=False,
#                           preserve_range=True),
#     FunctionWrapperDouble(create_dense_target, input=False, target=True),
#     FunctionWrapperDouble(np.moveaxis, input=True,
#                           target=False, source=-1, destination=0),
#     FunctionWrapperDouble(normalize_01)
# ])

# x_t, y_t = transforms(x, y)

# print(f'x = shape: {x.shape}; type: {x.dtype}')
# print(f'x = min: {x.min()}; max: {x.max()}')
# print(f'x_t: shape: {x_t.shape}  type: {x_t.dtype}')
# print(f'x_t = min: {x_t.min()}; max: {x_t.max()}')

# print(f'y = shape: {y.shape}; class: {np.unique(y)}')
# print(f'y_t = shape: {y_t.shape}; class: {np.unique(y_t)}')

import torch
from model import UNet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')
    
model = UNet(in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)

x = torch.randn(size=(1, 1, 512, 512), dtype=torch.float32)
with torch.no_grad():
    out = model(x)

from torchsummary import summary
summary = summary(model, (1, 512, 512),device="cpu")
