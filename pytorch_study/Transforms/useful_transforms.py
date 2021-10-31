from PIL import Image
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')

img = Image.open(r'E:\python_study\pytorch_study\小土堆教程\Transforms\dataset\train\bees_image\342758693_c56b89b6b6.jpg')
print(img)

#ToTensor()
img_tensor = transforms.ToTensor()(img)
print(type(img_tensor))
writer.add_image("tensor_input", img_tensor)
# writer.close()

#Normalize()
img_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_tensor)
writer.add_image("Normalize", img_norm)
# writer.close()

#Resize()
img_resize = transforms.Resize((512, 512))(img)
img_resize = transforms.ToTensor()(img_resize)
writer.add_image("Resize", img_resize)
# writer.close()

#Compose
trans_resize = transforms.Resize(512)
trans_totensor = transforms.ToTensor()

trans_compose = transforms.Compose([trans_resize, trans_totensor])
img_resize1 = trans_compose(img)

#RandomCrop()
trans_ramdom = transforms.RandomCrop(64)
trans_compose = transforms.Compose([trans_ramdom, trans_totensor])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("RandomCrop", img_crop, i)

