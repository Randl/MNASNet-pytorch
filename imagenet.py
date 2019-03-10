import argparse
import csv
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from tqdm import trange

import flops_benchmark
from MnasNet import MnasNet
from clr import CyclicLR
from cosine_with_warmup import CosineLR
from data import get_loaders
from logger import CsvLogger
from optimizer_wrapper import OptimizerWrapper
from run import train, test, save_checkpoint, find_bounds_clr

# https://arxiv.org/abs/1807.11626
# input_size, scale
claimed_acc_top1 = {224: {0.35: 0.624, 0.5: 0.678, 0.75: 0.715, 1.: 0.74, 1.3: 0.755, 1.4: 0.759}, 192: {1: 0.724},
                    160: {1: 0.707}, 128: {1: 0.673}, 96: {1: 0.623}}


def get_args():
    parser = argparse.ArgumentParser(description='MNASNet training with PyTorch')
    parser.add_argument('--dataroot', required=True, metavar='PATH',
                        help='Path to ImageNet train and val folders, preprocessed as described in '
                             'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='Number of data loading workers (default: 6)')
    parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

    # distributed
    parser.add_argument('--world-size', default=-1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='rank of distributed processes')
    parser.add_argument('--dist-init', default='env://', type=str, help='init used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    # Optimization options
    parser.add_argument('--sched', dest='sched', type=str, default='multistep')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The learning rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 300],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--step', type=int, default=40, help='Decrease learning rate each time.')
    parser.add_argument('--warmup', default=0, type=int, metavar='N', help='Warmup length')

    # CLR
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
    parser.add_argument('--max-lr', type=float, default=1, help='Maximal LR for CLR.')
    parser.add_argument('--epochs-per-step', type=int, default=20,
                        help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
    parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
    parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                        help='Run search for optimal LR in range (min_lr, max_lr)')

    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='Number of batches between log messages')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: random)')

    # Architecture
    parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MNASNet (default x1).')
    parser.add_argument('--input-size', type=int, default=224, metavar='I', help='Input size of MNASNet.')

    args = parser.parse_args()

    args.distributed = args.local_rank >= 0 or args.world_size > 1
    args.child = args.distributed and args.local_rank > 0
    if not args.distributed:
        args.local_rank = 0
        args.world_size = 1
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    args.save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(args.save_path) and not args.child:
        os.makedirs(args.save_path)

    if args.device == 'cuda' and torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True
        args.gpus = [args.local_rank]
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'

    if args.type == 'float64':
        args.dtype = torch.float64
    elif args.type == 'float32':
        args.dtype = torch.float32
    elif args.type == 'float16':
        args.dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    if not args.child:
        print("Random Seed: ", args.seed)
        print(args)
    return args


def is_bn(module):
    return isinstance(module, torch.nn.BatchNorm1d) or \
           isinstance(module, torch.nn.BatchNorm2d) or \
           isinstance(module, torch.nn.BatchNorm3d)


class Optimizer(object):
    def __init__(self):
        pass


def main():
    args = get_args()
    device, dtype = args.device, args.dtype

    model = MnasNet(width_mult=args.scaling)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    flops = flops_benchmark.count_flops(MnasNet, 1, device,
                                        dtype, args.input_size, 3, width_mult=args.scaling)
    if not args.child:
        print(model)
        print('number of parameters: {}'.format(num_parameters))
        print('FLOPs: {}'.format(flops))

    train_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, args.input_size,
                                           args.workers, args.world_size, args.local_rank)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    model, criterion = model.to(device=device, dtype=dtype), criterion.to(device=device, dtype=dtype)
    if args.dtype == torch.float16:
        for module in model.modules():  # FP batchnorm
            if is_bn(module):
                module.to(dtype=torch.float32)

    if args.distributed:
        args.device_ids = [args.local_rank]
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init, world_size=args.world_size,
                                rank=args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        print('Node #{}'.format(args.local_rank))
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer_class = torch.optim.SGD
    optimizer_params = {"lr": args.learning_rate, "momentum": args.momentum, "weight_decay": args.decay,
                        "nesterov": True}
    if args.find_clr:
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=True)
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=args.save_path)
        return

    if args.sched == 'clr':
        scheduler_class = CyclicLR
        scheduler_params = {"base_lr": args.min_lr, "max_lr": args.max_lr,
                            "step_size": args.epochs_per_step * len(train_loader), "mode": args.mode}
    elif args.sched == 'multistep':
        scheduler_class = MultiStepLR
        scheduler_params = {"milestones": args.schedule, "gamma": args.gamma}
    elif args.sched == 'cosine':
        scheduler_class = CosineLR
        scheduler_params = {"max_epochs": args.epochs, "warmup_epochs": args.warmup, "iter_in_epoch": len(train_loader)}
    elif args.sched == 'gamma':
        scheduler_class = StepLR
        scheduler_params = {"step_size": 30, "gamma": args.gamma}
    else:
        raise ValueError('Wrong scheduler!')

    optim = OptimizerWrapper(model, optimizer_class=optimizer_class, optimizer_params=optimizer_params,
                             scheduler_class=scheduler_class, scheduler_params=scheduler_params,
                             use_shadow_weights=args.dtype == torch.float16)
    best_test = 0

    # optionally resume from a checkpoint
    data = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint{}.pth.tar'.format(args.local_rank))
            csv_path = os.path.join(args.resume, 'results{}.csv'.format(args.local_rank))
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype, args.child)  # TODO
        return

    csv_logger = CsvLogger(filepath=args.save_path, data=data, local_rank=args.local_rank)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None
    if args.input_size in claimed_acc_top1:
        if args.scaling in claimed_acc_top1[args.input_size]:
            claimed_acc1 = claimed_acc_top1[args.input_size][args.scaling]
            if not args.child:
                csv_logger.write_text('Claimed accuracy is {:.2f}% top-1'.format(claimed_acc1 * 100.))
    train_network(args.start_epoch, args.epochs, optim, model, train_loader, val_loader, criterion,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, args.save_path, claimed_acc1,
                  claimed_acc5, best_test, args.local_rank, args.child)


def train_network(start_epoch, epochs, optim, model, train_loader, val_loader, criterion, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test, local_rank,
                  child):
    my_range = range if child else trange
    for epoch in my_range(start_epoch, epochs + 1):
        if not isinstance(optim.scheduler, CyclicLR) and not isinstance(optim.scheduler, CosineLR):
            optim.scheduler_step()
        train_loss, train_accuracy1, train_accuracy5, = train(model, train_loader, epoch, optim, criterion, device,
                                                              dtype, batch_size, log_interval, child)
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype, child)
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optim.state_dict()}, test_accuracy1 > best_test, filepath=save_path,
                        local_rank=local_rank)

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


if __name__ == '__main__':
    main()
