from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder
from pytorch_cinic.dataset import CINIC10
from typing import Optional,Callable,Tuple,Any

class OPENWORLDCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDCIFAR100, self).__init__(root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)
        
        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]

class OPENWORLDCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, labeled=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None, exist_label_list=[], clients_num=5):
        super(OPENWORLDCIFAR10, self).__init__(root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)
        
        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio, exist_label_list, clients_num)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio, exist_label_list, clients_num):
        labeled_idxs = []
        labeled_target = []
        unlabeled_idxs = []
        unlabeled_target = []
        # count_per_class = [0 for _ in range(10)]
        uniform_labeled_idx = []
        uniform_unlabeled_idx = []
        # labeled_num_per_class = [800, 800, 800, 800, 0, 0, 0, 0, 0, 0]
        # unlabeled_num_per_class = [800, 800, 800, 800, 1600, 1600, 1600, 1600, 1600, 1600]
        # 6
        labeled_num_per_class = [500, 500, 500, 500, 500, 500, 0, 0, 0, 0]
        unlabeled_num_per_class = [500, 500, 500, 500, 500, 500, 1000, 2500, 2500, 5000]
        # 8
        # labeled_num_per_class = [500, 500, 500, 500, 500, 500, 500, 500, 0, 0]
        # unlabeled_num_per_class = [500, 500, 500, 500, 500, 500, 500, 500, 1000, 1000]
        # 4 avg
        # labeled_num_per_class = [500, 500, 500, 500, 0, 0, 0, 0, 0, 0]
        # unlabeled_num_per_class = [500, 500, 500, 500, 1000, 5000, 5000, 5000, 5000, 5000]
        for idx, label in enumerate(self.targets):
            if label in exist_label_list:
                if label in labeled_classes and np.random.rand() < labeled_ratio:
                    labeled_idxs.append(idx)
                    labeled_target.append(label)
                    # indices = np.random.choice(len(labeled_idxs), len(labeled_idxs)//clients_num, False)
                    # arr_labeled_idxs = np.array(labeled_idxs)
                    # client_labeled_idxs = list(arr_labeled_idxs[indices])
                else:
                    unlabeled_idxs.append(idx)
                    unlabeled_target.append(label)
                    # indices = np.random.choice(len(unlabeled_idxs), len(labeled_idxs)//clients_num, False)
                    # arr_unlabeled_idxs = np.array(unlabeled_idxs)
                    # client_unlabeled_idxs = list(arr_unlabeled_idxs[indices])


        # indices = np.random.choice(len(labeled_idxs), len(labeled_idxs) // 3, False)
        # print("client_labeled num: ", len(labeled_idxs) // 3)
        arr_labeled_idxs = np.array(labeled_idxs)
        arr_labeled_target = np.array(labeled_target)
        #
        for i in range(10):
            idx = np.where(arr_labeled_target == i)[0]
            if len(idx) > 0:
                idx = np.random.choice(idx, labeled_num_per_class[i], False)
            # idx = np.random.choice(idx, label_per_class[i], False)
            uniform_labeled_idx.extend(idx)
        print("client_labeled num: ", len(uniform_labeled_idx))
        uniform_labeled_idx = np.array(uniform_labeled_idx)
        #
        client_labeled_idxs = list(arr_labeled_idxs[uniform_labeled_idx])

        # indices = np.random.choice(len(unlabeled_idxs), len(unlabeled_idxs) // 3, False)
        # print("client_unlabeled num: ", len(unlabeled_idxs) // 3)
        arr_unlabeled_idxs = np.array(unlabeled_idxs)
        arr_unlabeled_target = np.array(unlabeled_target)
        #
        for i in range(10):
            idx = np.where(arr_unlabeled_target == i)[0]
            # print("idx: ", idx)
            if len(idx) > 0:
                idx = np.random.choice(idx, unlabeled_num_per_class[i], False)
            # idx = np.random.choice(idx, label_per_class[i], False)
            uniform_unlabeled_idx.extend(idx)
        print("client_unlabeled num: ", len(uniform_unlabeled_idx))
        uniform_unlabeled_idx = np.array(uniform_unlabeled_idx)
        #
        client_unlabeled_idxs = list(arr_unlabeled_idxs[uniform_unlabeled_idx])

        return client_labeled_idxs, client_unlabeled_idxs  # labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        # for idx, label in enumerate(self.targets):
        #     if label > 5:  ## have to add because of small-classifier
        #         self.targets[idx] = 5
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]


class OPENWORLDCINIC10(CINIC10):

    def __init__(self, root, labeled=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None, exist_label_list=[], clients_num=5):
        super(OPENWORLDCINIC10, self).__init__(root, "train", transform, target_transform, download)

        self.partition = "train"
        # self.root = root

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.origin_data = ImageFolder(os.path.join(root, self.partition))
        self.data = []
        self.targets = []
        for data_target in self.origin_data:
            self.data.append(np.array(data_target[0]))
            self.targets.append(data_target[1])

        self.data = np.array(self.data)

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio,
                                                                            exist_label_list, clients_num)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_labeled_index(self, labeled_classes, labeled_ratio, exist_label_list, clients_num):
        labeled_idxs = []
        labeled_target = []
        unlabeled_idxs = []
        unlabeled_target = []
        # count_per_class = [0 for _ in range(10)]
        uniform_labeled_idx = []
        uniform_unlabeled_idx = []
        # labeled_num_per_class = [800, 800, 800, 800, 0, 0, 0, 0, 0, 0]
        # unlabeled_num_per_class = [800, 800, 800, 800, 1600, 1600, 1600, 1600, 1600, 1600]
        # 6
        labeled_num_per_class = [500, 500, 500, 500, 500, 500, 0, 0, 0, 0]
        unlabeled_num_per_class = [500, 500, 500, 500, 500, 500, 1000, 2500, 2500, 5000]
        # 4 avg
        # labeled_num_per_class = [500, 500, 500, 500, 0, 0, 0, 0, 0, 0]
        # unlabeled_num_per_class = [500, 500, 500, 500, 1000, 5000, 5000, 5000, 5000, 5000]
        for idx, label in enumerate(self.targets):
            if label in exist_label_list:
                if label in labeled_classes and np.random.rand() < labeled_ratio:
                    labeled_idxs.append(idx)
                    labeled_target.append(label)
                    # indices = np.random.choice(len(labeled_idxs), len(labeled_idxs)//clients_num, False)
                    # arr_labeled_idxs = np.array(labeled_idxs)
                    # client_labeled_idxs = list(arr_labeled_idxs[indices])
                else:
                    unlabeled_idxs.append(idx)
                    unlabeled_target.append(label)
                    # indices = np.random.choice(len(unlabeled_idxs), len(labeled_idxs)//clients_num, False)
                    # arr_unlabeled_idxs = np.array(unlabeled_idxs)
                    # client_unlabeled_idxs = list(arr_unlabeled_idxs[indices])

        # indices = np.random.choice(len(labeled_idxs), len(labeled_idxs) // 3, False)
        # print("client_labeled num: ", len(labeled_idxs) // 3)
        arr_labeled_idxs = np.array(labeled_idxs)
        arr_labeled_target = np.array(labeled_target)
        #
        for i in range(10):
            idx = np.where(arr_labeled_target == i)[0]
            if len(idx) > 0:
                idx = np.random.choice(idx, labeled_num_per_class[i], False)
            # idx = np.random.choice(idx, label_per_class[i], False)
            uniform_labeled_idx.extend(idx)
        print("client_labeled num: ", len(uniform_labeled_idx))
        uniform_labeled_idx = np.array(uniform_labeled_idx)
        #
        client_labeled_idxs = list(arr_labeled_idxs[uniform_labeled_idx])

        # indices = np.random.choice(len(unlabeled_idxs), len(unlabeled_idxs) // 3, False)
        # print("client_unlabeled num: ", len(unlabeled_idxs) // 3)
        arr_unlabeled_idxs = np.array(unlabeled_idxs)
        arr_unlabeled_target = np.array(unlabeled_target)
        #
        for i in range(10):
            idx = np.where(arr_unlabeled_target == i)[0]
            # print("idx: ", idx)
            if len(idx) > 0:
                idx = np.random.choice(idx, unlabeled_num_per_class[i], False)
            # idx = np.random.choice(idx, label_per_class[i], False)
            uniform_unlabeled_idx.extend(idx)
        print("client_unlabeled num: ", len(uniform_unlabeled_idx))
        uniform_unlabeled_idx = np.array(uniform_unlabeled_idx)
        #
        client_unlabeled_idxs = list(arr_unlabeled_idxs[uniform_unlabeled_idx])

        return client_labeled_idxs, client_unlabeled_idxs  # labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        # for idx, label in enumerate(self.targets):
        #     if label > 5:  ## have to add because of small-classifier
        #         self.targets[idx] = 5
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        #print("image size: ", self.data[0].size)
        # print("channel size: ", len(self.data[0].split()))
        tmp = np.array(self.data[0])
        #print("image to array shape: ", tmp.shape)
        self.data = self.data[idxs]
        #self.data = self.data[idxs, ...]


# Dictionary of transforms
dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
}
