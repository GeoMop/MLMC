import os
import re
import torch
import random
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import tensorflow as tf
import tensorflow_datasets as tfds
import imageio.v3 as iio
from PIL import Image


class ImageFlowDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, data_dir=None, config={}, version=None, feature_paths=None, target_paths=None, independent_samples=False,
                 transform=None, mean_target=None, std_target=None, mean_features=None, std_features=None, index=0):
        super(ImageFlowDataset, self).__init__()
        self._data_dir = data_dir

        self._index = index
        if config is None:
            config = {}
        self._config = config

        self._n_train_samples = self._config.get('n_train_samples', 2000)

        self.feature_paths = []
        self.target_paths = []
        self._transform = transform
        if target_paths is not None and feature_paths is not None:
            self.feature_paths = feature_paths
            self.target_paths = target_paths

        self._independent_samples = independent_samples

        self._dataset_config = {}
        if len(config) > 0:
            self._dataset_config = config.get('dataset_config', {})

        if self._data_dir is not None:
            self._set_path()

        self._mean_features = mean_features
        self._std_features = std_features
        self._mean_target = mean_target
        self._std_target = std_target

        if self._dataset_config.get("output_scale", False) and self._mean_target is None or self._std_target is None:
            print("get mean std target")
            self.get_mean_std_target(index=index, length=self._n_train_samples)

        if self._dataset_config.get("features_scale", False) and self._mean_features is None or self._std_features is None:
            self.get_mean_std_features(index=index, length=self._n_train_samples)

        self.signature = np.random.random()

    def dataset_config(self):
        if self._dataset_config is not None:
            return self._dataset_config
        return self._config.get('dataset_config', {})

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(256, 256, 3)),
                'label': tfds.features.ClassLabel(
                    names=['no', 'yes'],
                    doc='Whether this is a picture of a cat'),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""

        train_img_paths, train_tagret_paths = self.get_train_sample_paths(self._index, self._n_train_samples)
        test_img_paths, test_tagret_paths = self.get_test_sample_paths(self._index, self._n_train_samples)

        return {
            'train': self._generate_examples(features_paths=train_img_paths, target_paths=train_tagret_paths),
            'test': self._generate_examples(features_paths=test_img_paths, target_paths=test_tagret_paths),
        }

    def _generate_examples(self, features_paths=None, target_paths=None):
        # Read the input data out of the source files
        if features_paths is None:
            features_paths = self.feature_paths

        if target_paths is None:
            target_paths = self.target_paths

        self._dataset_config["mean_output"] = self._mean_target
        self._dataset_config["std_output"] = self._std_target
        self._dataset_config["mean_features"] = self._mean_features
        self._dataset_config["std_features"] = self._std_features
        self._save_data_config()

        for feature_path, target_path in zip(features_paths, target_paths):
            target = np.log(np.load(target_path))

            #print("target mean: {}, std: {}".format(self._mean_target, self._std_target))
            if self._mean_target is not None:
                target -= self._mean_target
            if self._std_target is not None:
                target /= self._std_target

            features = np.log(np.load(feature_path)["a"][..., np.newaxis])

            #print("features mean: {}, std: {}".format(self._mean_features, self._std_features))

            if self._mean_features is not None:
                features -= self._mean_features
            if self._std_features is not None:
                features /= self._std_features

            yield features, target

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
                if os.path.exists(os.path.join(sample_dir, self._config["file_name"])):
                    self.feature_paths.append(os.path.join(sample_dir, self._config["file_name"]))
                    self.target_paths.append(os.path.join(sample_dir, "output.npy"))

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature_path, target_path = self.feature_paths[idx], self.target_paths[idx]

        if isinstance(target_path, (list, np.ndarray)) and isinstance(feature_path, (list, np.ndarray)):
            return self._dataset_deepcopy(feature_path, target_path)

        target = np.log(np.load(target_path))

        # print("mean: {}, std: {}".format(self._mean_target, self._std_target))
        if self._mean_target is not None:
            target -= self._mean_target
        if self._std_target is not None:
            target /= self._std_target

        features = np.log(np.load(feature_path)["a"][..., np.newaxis])

        if self._mean_features is not None:
            features -= self._mean_features
        if self._std_features is not None:
            # print('features ', features)
            # print("std features ", self._std_features)
            features /= self._std_features

        # print("np.mean(features) ", np.mean(features))
        # print("np.std(features) ", np.std(features))

        return features, target

    def _plot_image(self, img):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 15))

        plt.imshow(img)
        plt.show()

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        temp = list(zip(self.feature_paths, self.target_paths))
        random.shuffle(temp)
        self.feature_paths, self.target_paths = zip(*temp)
        self.feature_paths, self.target_paths = list(self.feature_paths), list(self.target_paths)

    def get_train_sample_paths(self, index, length, **kwargs):
        feature_paths = self.feature_paths[index * length: index * length + length]
        target_paths = self.target_paths[index * length: index * length + length]

        return feature_paths, target_paths

    def _dataset_deepcopy(self, feature_paths, target_paths):
        new_dataset = copy.deepcopy(self)
        new_dataset._config = self._config
        new_dataset.feature_paths = feature_paths
        new_dataset.target_paths = target_paths
        new_dataset._mean_target = self._mean_target
        new_dataset._mean_features = self._mean_features
        new_dataset._std_target = self._std_target
        new_dataset._std_features = self._std_features

        return new_dataset

    def get_train_data(self, index, length):
        feature_paths = self.feature_paths[index * length: index * length + length]
        target_paths = self.target_paths[index * length: index * length + length]

        return self._dataset_deepcopy(feature_paths, target_paths)

    def get_test_sample_paths(self, index, length, **kwargs):
        if self._independent_samples:
            if index > 0:
                feature_paths = self.feature_paths[-index * length - length:-index * length]
                target_paths = self.target_paths[-index * length - length:-index * length]
            else:
                feature_paths = self.feature_paths[-index * length - length:]
                target_paths = self.target_paths[-index * length - length:]

        else:
            feature_paths = self.feature_paths[0:index * length] + self.feature_paths[index * length + length:]
            target_paths = self.target_paths[0:index * length] + self.target_paths[index * length + length:]

        return feature_paths, target_paths

    def get_test_data(self, index, length, **kwargs):
        if self._independent_samples:
            if index > 0:
                feature_paths = self.feature_paths[-index * length - length:-index * length]
                target_paths = self.target_paths[-index * length - length:-index * length]
            else:
                feature_paths = self.feature_paths[-index * length - length:]
                target_paths = self.target_paths[-index * length - length:]

        else:
            feature_paths = self.feature_paths[0:index * length] + self.feature_paths[index * length + length:]
            target_paths = self.target_paths[0:index * length] + self.target_paths[index * length + length:]
        return self._dataset_deepcopy(feature_paths, target_paths)

    def get_mean_std_features(self, index=0, length=None):
        if self._mean_features is None or self._std_features is None:
            if length is not None:
                feature_paths = self.feature_paths[index * length: index * length + length]
            else:
                feature_paths = self.feature_paths

            all_features = []
            for feature_path in feature_paths:
                all_features.extend(np.log(np.load(feature_path)["a"]))

                if len(all_features) > 25000:
                    break

            self._mean_features = np.mean(all_features)
            self._std_features = np.std(all_features)

        return self._mean_features, self._std_features

    def get_mean_std_target(self, index=0, length=None):
        if self._mean_target is None or self._std_target is None:
            if length is not None:
                target_paths = self.target_paths[index * length: index * length + length]
            else:
                target_paths = self.target_paths

            all_targets = []
            for target_path in target_paths:
                 all_targets.append(np.log(np.load(target_path)))

            self._mean_target = np.mean(all_targets, axis=0)
            self._std_target = np.std(all_targets, axis=0)

        return self._mean_target, self._std_target

    def _save_data_config(self):
        # Save config to Pickle
        import pickle
        import shutil

        if "iter_dir" in self._config:
            if os.path.exists(os.path.join(self._config['iter_dir'], "dataset_config.pkl")):
                os.remove(os.path.join(self._config['iter_dir'], "dataset_config.pkl"))

            # create a binary pickle file
            with open(os.path.join(self._config['iter_dir'], "dataset_config.pkl"), "wb") as writer:
                pickle.dump(self._dataset_config, writer)
