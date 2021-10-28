from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')
img_path = 'dataset/train/ants_image/6240338_93729615ec.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image('test', img_array, 1, dataformats='HWC')


for i in range(100):
    writer.add_scalar('y=2x', 2*i, i)

writer.close()








