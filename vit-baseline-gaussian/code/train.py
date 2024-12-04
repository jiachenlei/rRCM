# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log

import accelerate
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--path', type=str, default=None, help='path to pre-trained weights')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--use_mixup', action="store_true", default=False,
                    help='Whether use mixup')
parser.add_argument('--data_aug', action="store_true", default=False,
                    help='Whether use auto data augmentation')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--norm_stats', default="0.5", type=str,
                    help='statistics for normalize imagenet data')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def main():

    accelerator = accelerate.Accelerator(mixed_precision="no", split_batches=True)
    # if not os.path.exists(args.outdir):

    if accelerator.is_main_process:
        os.makedirs(args.outdir, exist_ok=True)

    train_dataset = get_dataset(args.dataset, 'train', autoaug=args.data_aug)
    test_dataset = get_dataset(args.dataset, 'test')

    if args.use_mixup:
        mixup_fn = Mixup(
                mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
                prob=1.0, switch_prob=0.5, mode="batch",
                label_smoothing=0.1, num_classes=1000)
    else:
        mixup_fn = None

    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset, normalize_stats=args.norm_stats) # nn.Module

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    log(logfilename, str(args))
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.path:
        ckpt = torch.load(args.path)
        optimizer.load_state_dict( ckpt["optimizer"] )
        new_state_dict = {}
        for k,v in ckpt["state_dict"].items():
            new_state_dict[k[len("module."):]] = v
        model.load_state_dict(new_state_dict)
        start_epoch = ckpt["epoch"]
        print(f"loaded pre-trained weights: {args.path}")
        print(f"start epoch: {start_epoch}")

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma, )
    print(scheduler.get_last_lr())
    train_loader, test_loader, model, optimizer = accelerator.prepare(train_loader, test_loader, model, optimizer)
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(accelerator, train_loader, model, criterion, optimizer, epoch, mixup_fn, args.noise_sd)
        test_acc = test(accelerator, test_loader, model, criterion, args.noise_sd)
        after = time.time()

        if accelerator.is_main_process:
            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))),
                float(scheduler.get_lr()[0]), float(train_loss), float(train_acc), float(test_acc)))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))


def train(accelerator, loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, mixup_fn, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    device = accelerator.device
    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs) * noise_sd

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        if mixup_fn is None:
            acc1 = accuracy(outputs, targets, topk=(1, ))[0]
            top1.update(acc1.item(), inputs.size(0))
            # top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if mixup_fn is None:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    return (losses.avg, top1.avg)


def test(accelerator, loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    device = accelerator.device
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.to(device)
            targets = targets.to(device)

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs) * noise_sd

            # compute output
            outputs = model(inputs)

            # measure accuracy and record loss
            # gather results from all processes
            shape = outputs.shape
            global_outputs = accelerator.gather(outputs).reshape(-1, shape[-1]) # Batch*Nodes, Class Num
            global_targets = accelerator.gather(targets).flatten() # Batch*Nodes
            # global_loss = accelerator.gather(loss).mean()

            if accelerator.is_main_process:
                acc1 = accuracy(global_outputs, global_targets, topk=(1, ))[0]
                # losses.update(global_loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(loader), batch_time=batch_time,
                        data_time=data_time, top1=top1))

        return top1.avg


if __name__ == "__main__":
    main()
