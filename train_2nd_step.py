"""
ref. from https://github.com/zxhuang1698/interpretability-by-parts
modified by sierkinhane
"""
import argparse
import random
import time

import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict as edict
from torchvision import transforms

from dataset import *
from models.loss import AreaLoss, WeightedEntropyLoss
from models.model import FineModel
from utils.func_utils import *
from utils.im_utils import *
from utils.log_utils import *

# benchmark before running
cudnn.benchmark = True


# arguments for the script itself
def parse_arg():
    parser = argparse.ArgumentParser(description="train image classification network")
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='config/CUB_200_2011.yaml')
    parser.add_argument('--experiment', type=str, required=True, help='save different experiments')
    parser.add_argument('--evaluate', type=str, default=False, help='evaluation mode')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)
    config.EXPERIMENT = args.experiment
    config.EVALUTATE = args.evaluate

    return config


# global variable for accuracy
best_acc = 0
best_core = 0
best_epoch = 0
best_core_threshold = 0
best_core_epoch = 0
best_top1_loc = 0
best_top5_loc = 0


def main():
    config = parse_arg()

    # fix all the randomness for reproducibility (for faster training and inference, please enable cudnn)
    # torch.backends.cudnn.enabled = False
    if config.SEED != -1:
        print('set seed!')
        config.SEED = 517
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)
        torch.backends.cudnn.deterministic = True

    # create folder for logging
    create_folders(config)

    # create model
    print("=> creating model...")
    model = FineModel(config.C_BASE, config.ATT_CHECKPOINT, config.CLS_CHECKPOINT,
                      num_classes=config.NUM_CLASSES).cuda()
    model_info(model)

    # define loss function (criterion) and optimizer
    criterion = [torch.nn.CrossEntropyLoss().cuda(), AreaLoss(topk=25).cuda(), WeightedEntropyLoss().cuda()]

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.R_LR)

    # data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # wrap to dataset
    train_data = CUB200(root=config.ROOT, train=True, transform=train_transforms)
    test_data = CUB200(root=config.ROOT, train=False, transform=test_transforms)
    print('load {} train images!'.format(len(train_data)))
    print('load {} test images!'.format(len(test_data)))

    # wrap to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.WORKERS, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.WORKERS, pin_memory=True)

    if config.EVALUTATE:
        print('Start evaluation...')
        checkpoint = torch.load('{}/checkpoints/{}/last_epoch.pth'.format(config.DEBUG, config.EXPERIMENT),
                                map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        best_threshold = test_multi_threshold(config, test_loader, model, criterion, 0)

        return

    num_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * config.R_EPOCHS)

    # optionally resume from a checkpoint (deprecated)
    start_epoch = 0
    if config.RESUME != '':
        checkpoint = torch.load(config.RESUME, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found")

    # save log
    sys.stdout = Logger(f'{config.DEBUG}/logs/{config.EXPERIMENT}_log.txt')

    # training part
    for epoch in range(start_epoch, config.R_EPOCHS):
        train(config, train_loader, model, criterion, optimizer, epoch, scheduler)

    best_threshold = test_multi_threshold(config, test_loader, model, criterion, epoch)
    torch.save(
        {"state_dict": model.state_dict(),
         "best_threshold": best_threshold,
         "optimizer": optimizer.state_dict(),
         "lr_scheduler": scheduler.state_dict(),
         }, '{}/checkpoints/{}/last_epoch.pth'.format(config.DEBUG, config.EXPERIMENT))

    print('Training finished.')


def train(config, train_loader, model, criterion, optimizer, epoch, scheduler):
    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_eval = AverageMeter()
    losses_area = AverageMeter()
    losses_entropy = AverageMeter()
    acc1s = AverageMeter()  # for evaluation accuracy

    # switch to train mode
    model.train()
    # record time
    end = time.time()

    # training step
    for i, (input, target, _, _) in enumerate(train_loader):

        # data to gpu
        input = input.cuda()
        target = target.cuda()

        # compute att_out
        main_out, att, att_dropout, features = model(input)

        loss_eval = criterion[0](main_out, target)  # for evaluation
        loss_area = criterion[1](att, main_out, features)
        loss_entropy = criterion[2](att)

        loss = loss_eval + 0.02 * loss_area + 1.5 * loss_entropy

        # record the losses and accuracy
        acc1 = accuracy(main_out.data, target)[0]
        acc1s.update(acc1.item(), input.size(0))
        losses.update(loss.data.item(), input.size(0))
        losses_eval.update(loss_eval.data.item(), input.size(0))
        losses_area.update(loss_area.data.item(), input.size(0))
        losses_entropy.update(loss_entropy.data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print the current status
        if i % config.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Eval {eval.val:.4f} ({eval.avg:.4f}) '
                  'Area {area.val:.4f} ({area.avg:.4f}) '
                  'Entropy {entropy.val:.4f} ({entropy.avg:.4f}) '
                  'Acc {acc1.val:.3f} ({acc1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, eval=losses_eval, area=losses_area, entropy=losses_entropy, acc1=acc1s), flush=True)

            # image debug
            image_debug(config.EXPERIMENT, input.clone().detach(), att_dropout, i)

    # print the learning rate
    lr = scheduler.get_last_lr()[0]
    print("Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))


def test_multi_threshold(config, test_loader, model, criterion, epoch):
    global best_core
    global best_core_epoch
    global best_core_threshold
    global best_top1_loc
    global best_top5_loc

    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_eval = AverageMeter()
    losses_area = AverageMeter()
    losses_entropy = AverageMeter()
    acc1s = AverageMeter()  # for evaluation accuracy

    # switch to evaluate mode
    model.eval()
    threshold = [(i + 1) / config.NUM_THRESHOLD for i in range(config.NUM_THRESHOLD - 1)]

    # record the time
    end = time.time()

    total = 0
    Lcorrect_1 = torch.Tensor([[0] for i in range(len(threshold))])
    Lcorrect_5 = torch.Tensor([[0] for i in range(len(threshold))])
    Corcorrect = torch.Tensor([[0] for i in range(len(threshold))])
    cnt = 0
    # testing
    with torch.no_grad():
        for i, (input, target, bboxes, cls_name, img_name) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()

            # inference the model
            main_out, att, att_dropout, features = model(input, inference=True)

            pred_boxes_t = np.zeros((len(threshold), input.size(0), 4))  # x0,y0, x1, y1

            for j in range(input.size(0)):

                estimated_boxes_at_each_thr = find_bbox(att[j, 0, :, :].detach().cpu().numpy().astype(np.float64),
                                                        threshold=threshold)
                for k in range(len(estimated_boxes_at_each_thr)):
                    pred_boxes_t[k, j, :] = estimated_boxes_at_each_thr[k]

            # compute the prediction loss
            loss_eval = criterion[0](main_out, target)  # for evaluation
            loss_area = criterion[1](att, main_out, features)
            loss_entropy = criterion[2](att)

            loss = loss_eval + 0.02 * loss_area + 1.5 * loss_entropy

            # record the losses and accuracy
            acc1 = accuracy(main_out.data, target)[0]
            acc1s.update(acc1.item(), input.size(0))
            losses.update(loss.data.item(), input.size(0))
            losses_eval.update(loss_eval.data.item(), input.size(0))
            losses_area.update(loss_area.data.item(), input.size(0))
            losses_entropy.update(loss_entropy.data.item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()

            # print(pred_boxes_t)
            # att_out = att_out[:,:200]
            total += input.size(0)
            for j in range(len(threshold)):
                pred_boxes = torch.from_numpy(pred_boxes_t[j]).float()
                gt_boxes = bboxes[:, 1:].float()

                # calculate
                inter = intersect(pred_boxes, gt_boxes)

                area_a = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union = area_a + area_b - inter
                IOU = inter / union
                IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
                IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

                _, pred = main_out.topk(5, 1, True, True)
                pred = pred.cpu()
                pred = pred.t()
                correct = pred.eq(target.cpu().view(1, -1).expand_as(pred))
                temp_1 = correct[:1, :].view(-1) * IOU.byte()
                temp_5 = torch.sum(correct[:5, :], 0).view(-1).byte() * IOU.byte()
                Lcorrect_1[j] += temp_1.sum()
                Lcorrect_5[j] += temp_5.sum()
                Corcorrect[j] += IOU.sum()

            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % config.PRINT_FREQ == 0:
                print('Test: [{0}][{1}/{2}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'Eval {eval.val:.4f} ({eval.avg:.4f}) '
                      'Area {area.val:.4f} ({area.avg:.4f}) '
                      'Entropy {entropy.val:.4f} ({entropy.avg:.4f}) '
                      'Acc {acc1.val:.3f} ({acc1.avg:.3f})'.format(
                    epoch, i, len(test_loader), batch_time=batch_time,
                    loss=losses, eval=losses_eval, area=losses_area, entropy=losses_entropy, acc1=acc1s), flush=True)

                # image debug
                visualization(config.EXPERIMENT, input.clone().detach(), att, cls_name, img_name,
                              phase='test', bboxes=pred_boxes_t[config.NUM_THRESHOLD // 2], gt_bboxes=bboxes)

            cnt += 1

    # print the accuracy after the testing
    print(' \033[92m* Accuracy: {acc.avg:.3f}\033[0m'.format(acc=acc1s))
    current_best_core = 0
    current_best_core_threshold = 0
    current_best_top1_loc = 0
    current_best_top5_loc = 0
    for i in range(len(threshold)):
        if (Corcorrect[i].item() / total) * 100 > current_best_core:
            current_best_core = (Corcorrect[i].item() / total) * 100
            current_best_core_threshold = threshold[i]
            current_best_top1_loc = (1 - Lcorrect_1[i].item() / float(total)) * 100
            current_best_top5_loc = (1 - Lcorrect_5[i].item() / float(total)) * 100
        if (Corcorrect[i].item() / total) * 100 > best_core:
            best_core = (Corcorrect[i].item() / total) * 100
            best_core_threshold = threshold[i]
            best_core_epoch = epoch
            best_top1_loc = (1 - Lcorrect_1[i].item() / float(total)) * 100
            best_top5_loc = (1 - Lcorrect_5[i].item() / float(total)) * 100

    print('  Best  => Correct: {:.2f}, threshold: {}, Top1 Loc: {:.2f}, Top5 Loc: {:.2f}'.format(best_core,
                                                                                                 best_core_threshold,
                                                                                                 best_top1_loc,
                                                                                                 best_top5_loc))
    print('Current => Correct: {:.2f}, threshold: {}, Top1 Loc: {:.2f}, Top5 Loc: {:.2f}'.format(current_best_core,
                                                                                                 current_best_core_threshold,
                                                                                                 current_best_top1_loc,
                                                                                                 current_best_top5_loc))

    return current_best_core_threshold


if __name__ == '__main__':
    main()
