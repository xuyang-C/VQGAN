import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
import torch
# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    # train_data = ImagePaths(args.dataset_path, size=128)
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    dataset = torchvision.datasets.ImageFolder(root=args.dataset_path,
                                               transform=torchvision.transforms.Compose([
                                                   #    torchvision.transforms.Resize((28,28)),
                                                   torchvision.transforms.CenterCrop((args.image_size, args.image_size)),
                                                   torchvision.transforms.ToTensor(),
                                                   # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader

class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path
def load_test_data(args):
    test_data = CustomImageFolder(root=args.dataset_path,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.CenterCrop((args.image_size, args.image_size)),
                                                   torchvision.transforms.ToTensor(),
                                               ]))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return test_loader

# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

class FrameDataset(Dataset):
    def __init__(self, args, transform=None):
        """
        root_dir: 包含所有图像文件夹的根目录
        transform: 应用于每帧图像的转换操作
        """
        self.root_dir = args.dataset_path
        self.transform = transform
        self.frame_groups = []

        # 遍历文件夹，获取图像组
        # 使用os.walk遍历所有子目录
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # 过滤出JPG图像
            frames = sorted([os.path.join(dirpath, f) for f in filenames if f.endswith('.jpg')])

            # 创建连续的7帧图像组
            for i in range(len(frames) - args.num_frames):
                self.frame_groups.append(frames[i:i + args.num_frames+1])

    def __len__(self):
        return len(self.frame_groups)

    def __getitem__(self, idx):
        group = self.frame_groups[idx]
        images = [Image.open(img_path).convert('RGB') for img_path in group]

        if self.transform:
            images = [self.transform(image) for image in images]

        # 将图像列表堆叠成一个Tensor
        images = torch.stack(images)
        return images

def load_video_data(args):
    # 创建数据集实例
    dataset = FrameDataset(args, transform=torchvision.transforms.Compose([torchvision.transforms.CenterCrop((args.image_size, args.image_size)),
    torchvision.transforms.ToTensor(),]))
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return data_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
