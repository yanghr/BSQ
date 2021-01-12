'''
 python cifar_prune_STE.py -a resnet --depth 20 --epochs 350 --schedule 250 --gamma 0.1 --wd 1e-4 --model checkpoints/cifar10/resnet-20-8/model_best.pth.tar --decay 0.002 --Prun_Int 50 --thre 0.0 --checkpoint checkpoints/cifar10/xxx --Nbits 8 --act 4 --bin --L1 >xxx.txt
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from models.cifar.bit import BitLinear, BitConv2d

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import util


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=350, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--Nbits', default=4, type=int, metavar='N',
                    help='Number of bits in conv layer')
parser.add_argument('--act', default=4, type=int, metavar='N',
                    help='Activation precision')
parser.add_argument('--bin', action='store_true', default=False,
                    help='Use binary format of the model')
parser.add_argument('--Prun_Int', default=50, type=int, metavar='N',
                    help='Interval between pruning is performed')
parser.add_argument('--thre', default=0.0, type=float, metavar='N',
                    help='Pruning threshold')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--decay', type=float, default=0.01, metavar='D',
                    help='decay for bit-sparse regularizer (default: 0.01)')
parser.add_argument('--L1', action='store_true', default=False,
                    help='Use L1 regularizer')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', type=str, default=None,
                    help='log file name')   
                    
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed',default=1234, type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                    Nbits = args.Nbits,
                    act_bit = args.act,
                    bin = args.bin
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    #util.print_model_parameters(model)
    
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    # Resume
    if args.model is not None:
        print('loading pretrained model')
        checkpoint = torch.load(args.model)
        if 'Nbit_dict' in checkpoint:
            model.module.set_Nbits(checkpoint['Nbit_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        
        if not args.bin:
            for name, module in model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    module.to_bin()
        
        for name, module in model.named_modules():
            if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                print(name)
                module.print_stat()
        test_loss, accuracy = test(testloader, model, criterion, 0, use_cuda)
        print(f'Accuracy: {accuracy:.2f}%')
        
    Nbit_dict = model.module.pruning(threshold=0.0, drop=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    TP = model.module.total_param()
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Reg Loss', 'Comp'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        
        if args.Prun_Int:
            if epoch % args.Prun_Int==0:
                for name, module in model.named_modules():
                    if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                        module.quant(maxbit = 8) #args.Nbits)
                Nbit_dict = model.module.pruning(threshold=args.thre, drop=True)
                print('########Model after pruning########')
                for name, module in model.named_modules():
                    if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                        print(name)
                        module.print_stat()
                test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
                print(' Test Loss after pruning:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
                ## Reset optimizer to fit changed variable dimension 
                del optimizer
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
                TP = model.module.total_param()
                TB = model.module.total_bit()
                Comp = (TP*32)/TB
                print(' Compression rate after pruning [%d / %d]:  %.2f X' % (TP*32, TB, Comp))

        train_loss, train_acc, reg_loss = train(trainloader, model, criterion, optimizer, epoch, use_cuda, TP, Comp)
        
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        print('Total Loss: {test_loss:.4f} | top1: {test_acc: .4f}'.format(
                    test_loss=test_loss,
                    test_acc=test_acc,
                    ))

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, reg_loss, Comp])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'Nbit_dict': Nbit_dict,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    #logger.plot()
    #savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Final checkpoint:')
    for name, module in model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit = 8)
            
    Nbit_dict = model.module.pruning(threshold=0.0)
    
    for name, module in model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            print(name)
            module.print_stat()
    
    TP = model.module.total_param()
    TB = model.module.total_bit()
    Comp = (TP*32)/TB
    print(' Final compression rate [%d / %d]:  %.2f X' % (TP*32, TB, Comp))
    
    test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
    print('Total Loss: {test_loss:.4f} | top1: {test_acc: .4f}'.format(
                    test_loss=test_loss,
                    test_acc=test_acc,
                    ))
    save_checkpoint({
                'Nbit_dict': Nbit_dict,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, False, checkpoint=args.checkpoint)

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, TP, Comp):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Tlosses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        reg=0.
        if args.decay:
            for name, module in model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    if args.L1:
                        reg = module.L1reg(reg)
                    else:
                        reg = 0.

        total_loss = loss+args.decay*reg/TP
        treg = reg.item()/TP      

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        Tlosses.update(total_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if args.bin:
            for name, p in model.named_parameters():
                if 'scale' in name or 'bn' in name:
                    continue
                else:
                    tensor = p.data    
                    p.data = torch.where(tensor > 2, torch.full_like(p.data, 2), p.data)
                    p.data = torch.where(tensor < 0, torch.full_like(p.data, 0), p.data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '(CP: {Comp:.2f}X | Epoch:{epoch} {batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Total Loss: {tloss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    Comp=Comp,
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    tloss=Tlosses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, treg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
