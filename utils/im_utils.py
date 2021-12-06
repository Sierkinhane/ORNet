# system lib
import os

import cv2
import matplotlib
# pytorch lib
import torch
from torchvision import utils
from torchvision.utils import make_grid

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as st


def image_debug(experiments, images, attmaps, i, phase='train', nrwos=4):
    """
    debug during training
    """
    # save images in a batch
    utils.save_image(images, 'debug/images/{}/{}/epoch-{}-raw.jpg'.format(experiments, phase, i), nrow=nrwos,
                     normalize=True)

    attmaps = attmaps.detach().squeeze().to('cpu').numpy()

    # visualize attention map
    fig, axes = plt.subplots(nrwos, nrwos, figsize=(21, 21))
    for j in range(axes.shape[0]):
        for k in range(axes.shape[1]):
            temp = attmaps[j * nrwos + k, :, :]
            axes[j, k].imshow(temp)
            axes[j, k].axis('off')
    plt.savefig('debug/images/{}/{}/epoch-{}-att.jpg'.format(experiments, phase, i), bbox_inches='tight')
    plt.close()

    # visualize the histogram of attention map
    fig, axes = plt.subplots(nrwos, nrwos, figsize=(21, 21))
    kde_xs = np.linspace(0, 1, 300)
    for j in range(axes.shape[0]):
        for k in range(axes.shape[1]):
            temp = attmaps[j * nrwos + k, :, :]
            axes[j, k].hist(temp.ravel(), bins=50, density=True, range=(0, 1), ec='black', fc='pink')
            kde = st.gaussian_kde(temp.ravel())
            axes[j, k].plot(kde_xs, kde.pdf(kde_xs), c='purple', linewidth=2.5)
            # axes[j, k].axis('off')
    plt.savefig('debug/images/{}/{}/epoch-{}-hist.jpg'.format(experiments, phase, i), bbox_inches='tight')
    plt.close()


import cmapy


def visualization(experiments, images, attmaps, cls_name, image_name, phase='train', bboxes=None, gt_bboxes=None):
    _, c, h, w = images.shape
    attmaps = attmaps.detach().squeeze().to('cpu').numpy()

    for i in range(images.shape[0]):

        # create folder
        if not os.path.exists('debug/images/{}/{}/{}'.format(experiments, phase, cls_name[i])):
            os.mkdir('debug/images/{}/{}/{}'.format(experiments, phase, cls_name[i]))

        attmap = attmaps[i]
        attmap = attmap / np.max(attmap)
        attmap = np.uint8(attmap * 255)

        colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cmapy.cmap('seismic'))

        grid = make_grid(images[i].unsqueeze(0), nrow=1, padding=0, pad_value=0,
                         normalize=True, range=None)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]

        cam = colormap + 0.5 * image
        cam = cam / np.max(cam)
        cam = np.uint8(cam * 255).copy()
        bbox_image = image.copy()

        if bboxes is not None:
            bbox = None
            if gt_bboxes is not None:
                if isinstance(gt_bboxes, list):
                    bbox = bboxes[i][0]
                    for j in range(gt_bboxes[i].shape[0]):
                        gtbox = gt_bboxes[i][j]
                        cv2.rectangle(bbox_image, (int(gtbox[1]), int(gtbox[2])), (int(gtbox[3]), int(gtbox[4])),
                                      (255, 0, 0), 2)
                else:
                    bbox = bboxes[i]
                    gtbox = gt_bboxes[i]
                    cv2.rectangle(bbox_image, (int(gtbox[1]), int(gtbox[2])), (int(gtbox[3]), int(gtbox[4])),
                                  (255, 0, 0), 2)
            cv2.rectangle(bbox_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)  # BGR

        cv2.imwrite(f'debug/images/{experiments}/{phase}/{cls_name[i]}/{image_name[i]}_raw.jpg', image)
        cv2.imwrite(f'debug/images/{experiments}/{phase}/{cls_name[i]}/{image_name[i]}_bbox.jpg', bbox_image)
        cv2.imwrite(f'debug/images/{experiments}/{phase}/{cls_name[i]}/{image_name[i]}_att.jpg', cam)


