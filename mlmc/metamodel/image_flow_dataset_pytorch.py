import os
import re
import torch
import random
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import imageio.v3 as iio
from PIL import Image


class ImageFlowDataset(Dataset):

    def __init__(self, data_dir=None, config={}, image_paths=None, target_paths=None, independent_samples=False,
                 transform=None, mean_target=None, std_target=None):
        super(ImageFlowDataset, self).__init__()
        self._data_dir = data_dir

        self.image_paths = []
        self.target_paths = []
        self._transform = transform
        if target_paths is not None and image_paths is not None:
            self.image_paths = image_paths
            self.target_paths = target_paths

        self._independent_samples = independent_samples

        if config is not None:
            self._dataset_config = config.get('dataset_config', {})

        if self._data_dir is not None:
            self._set_path()

        self._mean_target = mean_target
        self._std_target = std_target

        self.signature = np.random.random()

    def _set_path(self):
        if self._data_dir is None:
            raise AttributeError

        for idx, s_dir in enumerate(os.listdir(self._data_dir)):
            try:
                l = re.findall(r'L(\d+)_S', s_dir)[0]
            except IndexError:
                    continue
            if os.path.isdir(os.path.join(self._data_dir, s_dir)):
                sample_dir = os.path.join(self._data_dir, s_dir)
                if os.path.exists(os.path.join(sample_dir, "bypixel_512.npz")):
                    self.image_paths.append(os.path.join(sample_dir, "bypixel_512.npz"))
                    self.target_paths.append(os.path.join(sample_dir, "output.npy"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #print("target paths ", self.target_paths)
        img_path, target_path = self.image_paths[idx], self.target_paths[idx]

        #("target path ", target_path)

        if isinstance(target_path, (list, np.ndarray)) and isinstance(img_path, (list, np.ndarray)):
            new_dataset= copy.deepcopy(self)
            new_dataset.image_paths = img_path
            new_dataset.target_paths = target_path
            return new_dataset
        #img = iio.imread(img_path)[..., :3]  # image in RGB
        target = np.load(target_path)

        # print("mean: {}, std: {}".format(self._mean_target, self._std_target))
        # print("target ", target)
        if self._mean_target is not None:
            target -= self._mean_target
        if self._std_target is not None:
            target /= self._std_target
        #print("normalized target ", target)

        if self._transform is None:
            tf = transforms.Compose([
                #iio.imread(img_path)[..., :3],  # str path -> img data
                #transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                #transforms.RandomRotation(15),
                #transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # from imagenet
            ])
        else:
            tf = self._transform

        #self._plot_image(iio.imread(img_path)[..., :3])
        img = tf(np.load(img_path)["a"])#tf((iio.imread(img_path)[..., :3]))
        label = torch.tensor(target)

        return img, label

    def _plot_image(self, img):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 15))

        plt.imshow(img)
        plt.show()

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        temp = list(zip(self.image_paths, self.target_paths))
        random.shuffle(temp)
        self.image_paths, self.target_paths = zip(*temp)
        self.image_paths, self.target_paths = list(self.image_paths), list(self.target_paths)

    def get_train_data(self, index, length, **kwargs):
        image_paths = self.image_paths[index * length: index * length + length]
        target_paths = self.target_paths[index * length: index * length + length]

        return ImageFlowDataset(image_paths=image_paths, target_paths=target_paths, **kwargs)

    def get_test_data(self, index, length, **kwargs):
        if self._independent_samples:
            if index > 0:
                image_paths =self.image_paths[-index * length - length:-index * length]
                target_paths = self.target_paths[-index * length - length:-index * length]
            else:
                image_paths = self.image_paths[-index * length - length:]
                target_paths = self.target_paths[-index * length - length:]

        else:
            image_paths = self.image_paths[0:index * length] + self.image_paths[index * length + length:]
            target_paths = self.target_paths[0:index * length] + self.target_paths[index * length + length:]

        return ImageFlowDataset(image_paths=image_paths, target_paths=target_paths, **kwargs)

    def get_mean_std_target(self, index=0, length=None):
        if self._mean_target is None or self._std_target is None:
            if length is not None:
                target_paths = self.target_paths[index * length: index * length + length]
            else:
                target_paths = self.target_paths

            all_targets = []
            for target_path in target_paths:
                 all_targets.append(np.load(target_path))

            self._mean_target = np.mean(all_targets, axis=0)
            self._std_target = np.var(all_targets, axis=0)

        return self._mean_target, self._std_target
