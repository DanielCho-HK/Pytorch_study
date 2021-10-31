from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = r'E:\python_study\pytorch_study\小土堆教程\Transforms\dataset\train\bees_image\36900412_92b81831ad.jpg'
img = Image.open(img_path)


writer = SummaryWriter('logs')


print(img)
img_tensor = transforms.ToTensor()(img)
print(img_tensor)

writer.add_image("tensor_img", img_tensor)
writer.close()