import numpy as np
import torch
import cv2


def find_bbox(scoremap, threshold=0.5, scale=4):
    if isinstance(threshold, list):
        bboxes = []
        for i in range(len(threshold)):
            indices = np.where(scoremap > threshold[i])
            try:
                miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
                maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
            except:
                bboxes.append([0, 0, 224, 224])
            else:
                bboxes.append([minx, miny, maxx, maxy])
        return bboxes

    else:
        indices = np.where(scoremap > threshold)
        try:
            miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
            maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
        except:
            return [0, 0, 224, 224]
        else:
            return [minx, miny, maxx, maxy]


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list, multi_contour_eval=False, scale=4):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0 * scale, y0 * scale, x1 * scale, y1 * scale])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


def normalize_scoremap(alm):
    """
    Args:
        alm: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(alm).any():
        return np.zeros_like(alm)
    if alm.min() == alm.max():
        return np.zeros_like(alm)
    alm -= alm.min()
    alm /= alm.max()
    return alm


def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def IOUFunciton_ILSRVC(boxes_a, boxes_b):
    IOUList = np.zeros(len(boxes_b))
    for bbox_i in range(len(boxes_b)):  # #image
        box_a = boxes_a[bbox_i][0]
        box_a = torch.from_numpy(box_a).float()
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        imgBoxes_b = boxes_b[bbox_i]
        tempIOU = 0
        for bbox_j in range(imgBoxes_b.shape[0]):  # #boxes in one image
            box_b = imgBoxes_b[bbox_j].float()
            # print(box_b)
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            intersect = (min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * (
                    min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
            abIOU = intersect / (area_a + area_b - intersect)
            if abIOU > tempIOU:
                tempIOU = abIOU
        IOUList[bbox_i] = tempIOU
    return torch.tensor(IOUList, dtype=torch.float)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the acc@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_binary(output, target):
    """Computes the accuracy for binary classification"""
    batch_size = target.size(0)
    pred = (output > 0.5).long()
    acc = (pred.squeeze(1) == target).float().mean()
    return acc * 100
