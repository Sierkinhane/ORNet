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
from models.evaluator import Evaluator
from utils.im_utils import *
from utils.log_utils import *

# benchmark before running
cudnn.benchmark = True


# arguments for the script itself
def parse_arg():
    parser = argparse.ArgumentParser(description="train image classification network")
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='config/CUB_200_2011.yaml')
    parser.add_argument('--experiment', type=str, required=True, help='save different experiments')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)
    config.EXPERIMENT = args.experiment

    return config


# global variable for accuracy
best_acc = 0


def main():
    global best_acc

    config = parse_arg()

    # fix all the randomness for reproducibility (for faster training and inference, please enable cudnn)
    # torch.backends.cudnn.enabled = False
    if config.SEED != -1:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    # create folder for logging
    if not os.path.exists(config.DEBUG):
        os.mkdir(config.DEBUG)
        os.mkdir('{}/checkpoints'.format(config.DEBUG))
    if not os.path.exists('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT)):
        os.mkdir('{}/checkpoints/{}'.format(config.DEBUG, config.EXPERIMENT))

    # create evaluator model
    print("=> creating model...")
    model = Evaluator(num_classes=config.NUM_CLASSES).cuda()
    model_info(model)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY,
                                momentum=config.MOMENTUM)

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

    num_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * config.EPOCHS)

    # optionally resume from a checkpoint
    start_epoch = 0
    if config.RESUME != '':
        checkpoint = torch.load(config.RESUME, map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
        print('best epoch: {}, best acc'.format(best_epoch, best_acc))
    else:
        print("=> no checkpoint found")

    # save log
    sys.stdout = Logger(f'{config.DEBUG}/logs/{config.EXPERIMENT}_log.txt')

    # training part
    for epoch in range(start_epoch, config.EPOCHS):

        # training
        train(config, train_loader, model, criterion, optimizer, epoch, scheduler)

        # evaluate on test set
        acc = test(config, test_loader, model, criterion)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            best_epoch = epoch
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "best_epoch": best_epoch,
                 "best_acc": best_acc,
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": scheduler.state_dict(),
                 }, '{}/checkpoints/{}/best_epoch.pth'.format(config.DEBUG, config.EXPERIMENT))
        else:
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "best_epoch": best_epoch,
                 "best_acc": best_acc,
                 "optimizer": optimizer.state_dict(),
                 "lr_scheduler": scheduler.state_dict(),
                 }, '{}/checkpoints/{}/current_epoch.pth'.format(config.DEBUG, config.EXPERIMENT))

        # print current best accuracy
        print('Current best average accuracy is: %.4f' % best_acc)

    # print the overall best acc and close the writer
    print('Training finished...')
    print('Best accuracy on test set is: %.4f.' % best_acc)


def train(config, train_loader, model, criterion, optimizer, epoch, scheduler):
    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    # switch to train mode
    model.train()
    # record time
    end = time.time()

    # training step
    for i, (input, target, _, _) in enumerate(train_loader):

        # data to gpu
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)

        # compute the prediction loss
        loss = criterion(output, target)

        # record the losses and accuracy
        acc = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1_accs.update(acc[0].item(), input.size(0))
        top5_accs.update(acc[1].item(), input.size(0))

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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Cls_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top1 Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Top5 Acc {acc2.val:.3f} ({acc2.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, acc=top1_accs, acc2=top5_accs), flush=True)

    # print the learning rate
    lr = scheduler.get_last_lr()[0]
    print("Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))


def test(config, test_loader, model, criterion):
    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    # testing
    with torch.no_grad():
        for i, (input, target, _, _, _) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()

            # inference the model
            output = model(input)

            # compute the prediction loss
            loss = criterion(output, target)

            # record the losses and accuracy
            acc = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))

            top1_accs.update(acc[0].item(), input.size(0))
            top5_accs.update(acc[1].item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % config.PRINT_FREQ == 0:
                print('Test [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Cls_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top1 Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                      'Top5 Acc {acc2.val:.3f} ({acc2.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, acc=top1_accs, acc2=top5_accs), flush=True)

    # print the accuracy after the testing
    print(' \033[92m* Top1 Accuracy: {acc.avg:.3f}\033[0m'.format(acc=top1_accs))
    print(' \033[92m* Top5 Accuracy: {acc.avg:.3f}\033[0m'.format(acc=top5_accs))

    # return the accuracy
    return top1_accs.avg


if __name__ == '__main__':
    main()
