from torch.utils.data import Dataset
from PIL import Image
import os

# img_path = 'E:\\BaiduNetdiskDownload\\hymenoptera_data\\train\\bees\\95238259_98470c5b10.jpg'
dir_path = 'E:\\BaiduNetdiskDownload\\hymenoptera_data\\train\\bees\\'
img_path_list = os.listdir(dir_path)

# img = Image.open(img_path)
# print(type(img))
# print(img.size)
# img.show()

root_dir = 'E:/BaiduNetdiskDownload/hymenoptera_data/train'
label_dir = 'bees'
# path = os.path.join(root_dir, label_dir)

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)



data = MyData(root_dir, label_dir)
img, label = data[0]
print(len(data))








