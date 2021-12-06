import xml.etree.ElementTree as ET

import PIL.Image
import torch
import torch.utils.data


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            lineSplit = line.strip().split(' ')
            imgPath, label = lineSplit[0], lineSplit[1]
            flag = lineSplit[2]
            imgList.append((imgPath, int(label), str(flag)))

    return imgList


def bboxes_reader(path):
    bboxes_list = {}
    bboxes_file = open(path + "/val.txt")
    for line in bboxes_file:
        line = line.split('\n')[0]
        line = line.split(' ')[0]
        labelIndex = line
        line = line.split("/")[-1]
        line = line.split(".")[0] + ".xml"
        bbox_path = path + "/val_boxes/val/" + line
        tree = ET.ElementTree(file=bbox_path)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        bbox_line = []
        for Object in ObjectSet:
            BndBox = Object.find('bndbox')
            xmin = BndBox.find('xmin').text
            ymin = BndBox.find('ymin').text
            xmax = BndBox.find('xmax').text
            ymax = BndBox.find('ymax').text
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            bbox_line.append([xmin, ymin, xmax, ymax])
        bboxes_list[labelIndex] = bbox_line
    return bboxes_list


class ILSVRC2012(torch.utils.data.Dataset):
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

    def __init__(self, root, train=True, transform=None):
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

        if self._train:
            self.imgList = default_list_reader(self._root + '/train.txt')[:1000]
        else:
            self.imgList = default_list_reader(self._root + '/val.txt')[:1000]

        self.bboxes = bboxes_reader(self._root)

    def __getitem__(self, index):

        img_name, label, cls_name = self.imgList[index]
        image = self.loader(self._root + '/' + img_name)

        newBboxes = []
        if not self._train:
            bboxes = self.bboxes[img_name]
            for bbox_i in range(len(bboxes)):
                bbox = bboxes[bbox_i]
                bbox[0] = bbox[0] * (256 / image.size[0]) - 16
                bbox[1] = bbox[1] * (256 / image.size[1]) - 16
                bbox[2] = bbox[2] * (256 / image.size[0]) - 16
                bbox[3] = bbox[3] * (256 / image.size[1]) - 16
                bbox.insert(0, index)
                newBboxes.append(bbox)

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)

        if self._train:
            return image, label, cls_name, img_name.split('/')[-1].split('.')[0]
        else:
            return image, label, newBboxes, cls_name, img_name.split('/')[-1].split('.')[0]  # only image name

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.imgList)


def my_collate(batch):
    images = []
    labels = []
    bboxes = []
    cls_name = []
    img_name = []
    for sample in batch:
        images.append(sample[0])
        labels.append(torch.tensor(sample[1]))
        bboxes.append(torch.FloatTensor(sample[2]))
        cls_name.append(sample[3])
        img_name.append(sample[4])

    return torch.stack(images, 0), torch.stack(labels, 0), bboxes, cls_name, img_name
