import torch

#方式1 -> 保存方式1，加載模型
import torchvision

model = torch.load('./vgg16_method1.pth')
# print(model)
#方式2 -> 保存方式2，加載模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16)

