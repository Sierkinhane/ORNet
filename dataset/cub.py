"""
ref. from https://github.com/zxhuang1698/interpretability-by-parts
modified by sierkinhane
"""

import os

import PIL.Image
import numpy as np
import torch
import torch.utils.data


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


class CUB200(torch.utils.data.Dataset):
    """
    CUB200 dataset.

    Variables
    ----------
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _train_data, list of np.array.
        _train_labels, list of int.
        _train_parts, list np.array.
        _train_boxes, list np.array.
        _test_data, list of np.array.
        _test_labels, list of int.
        _test_parts, list np.array.
        _test_boxes, list np.array.
    """

    def __init__(self, root, train=True, transform=None, resize=256):
        """
        Load the dataset.

        Args
        ----------
        root: str
            Root directory of the dataset.
        train: bool
            train/test data split.
        transform: callable
            A function/transform that takes in a PIL.Image and transforms it.
        resize: int
            Length of the shortest of edge of the resized image. Used for transforming landmarks and bounding boxes.

        """
        self._root = root
        self._train = train
        self._transform = transform
        self.loader = pil_loader
        self.newsize = resize

        # 15 key points provided by CUB
        self.num_kps = 15
        if not os.path.isdir(root):
            os.mkdir(root)

        # Load all data into memory for best IO efficiency. This might take a while
        if self._train:
            self._train_data, self._train_labels, self._train_parts, self._train_boxes = self._get_file_list(train=True)
            assert (len(self._train_data) == 5994
                    and len(self._train_labels) == 5994)
        else:
            self._test_data, self._test_labels, self._test_parts, self._test_boxes = self._get_file_list(train=False)
            assert (len(self._test_data) == 5794
                    and len(self._test_labels) == 5794)

    def __getitem__(self, index):
        """
        Retrieve data samples.

        Args
        ----------
        index: int
            Index of the sample.

        Returns
        ----------
        image: torch.FloatTensor, [3, H, W]
            Image of the given index.
        target: int
            Label of the given index.
        parts: torch.FloatTensor, [15, 4]
            Landmark annotations.
        boxes: torch.FloatTensor, [5, ]
            Bounding box annotations.
        """

        # load the variables according to the current index and split
        if self._train:
            image_path = self._train_data[index]
            target = self._train_labels[index]
            boxes = self._train_boxes[index]

        else:
            image_path = self._test_data[index]
            target = self._test_labels[index]
            boxes = self._test_boxes[index]

        # load the image
        image = self.loader(image_path)
        image = np.array(image)

        # numpy arrays to pytorch tensors
        boxes = torch.from_numpy(boxes).float()

        # xywh to x1y1 x2y2
        boxes[3] = boxes[1] + boxes[3]
        boxes[4] = boxes[2] + boxes[4]

        # for center crop
        boxes[1] = boxes[1] * (256 / image.shape[1]) - 16
        boxes[2] = boxes[2] * (256 / image.shape[0]) - 16
        boxes[3] = boxes[3] * (256 / image.shape[1]) - 16
        boxes[4] = boxes[4] * (256 / image.shape[0]) - 16

        # convert the image into a PIL image for transformation
        image = PIL.Image.fromarray(image)

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)

        # get the cls name and image name for visualization
        cls_name = image_path.split('/')[-2]
        img_name = image_path.split('/')[-1].split('.')[0]

        if self._train:
            return image, target, cls_name, img_name
        else:
            return image, target, boxes, cls_name, img_name

    def __len__(self):
        """Return the length of the dataset."""
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _get_file_list(self, train=True):
        """Prepare the data for train/test split and save onto disk."""

        # load the list into numpy arrays
        image_path = self._root + '/images/'
        id2name = np.genfromtxt(self._root + '/images.txt', dtype=str)
        id2train = np.genfromtxt(self._root + '/train_test_split.txt', dtype=int)
        id2part = np.genfromtxt(self._root + '/parts/part_locs.txt', dtype=float)
        id2box = np.genfromtxt(self._root + '/bounding_boxes.txt', dtype=float)

        # creat empty lists
        train_data = []
        train_labels = []
        train_parts = []
        train_boxes = []
        test_data = []
        test_labels = []
        test_parts = []
        test_boxes = []

        # iterating all samples in the whole dataset
        for id_ in range(id2name.shape[0]):

            # load each variable
            image = os.path.join(image_path, id2name[id_, 1])
            # Label starts with 0
            label = int(id2name[id_, 1][:3]) - 1
            parts = id2part[id_ * self.num_kps: id_ * self.num_kps + self.num_kps][:, 1:]
            boxes = id2box[id_]

            # training split
            if id2train[id_, 1] == 1:
                train_data.append(image)
                train_labels.append(label)
                train_parts.append(parts)
                train_boxes.append(boxes)
            # testing split
            else:
                test_data.append(image)
                test_labels.append(label)
                test_parts.append(parts)
                test_boxes.append(boxes)

        # return accoring to different splits
        if train:
            return train_data, train_labels, train_parts, train_boxes
        else:
            return test_data, test_labels, test_parts, test_boxes
