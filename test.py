import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from mmseg.apis import MMSegInferencer
inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
inferencer('demo/demo.png', show=True)