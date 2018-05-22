import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from utils import progress_bar
from models import HiResA
from torch.optim.lr_scheduler import MultiStepLR

try_no = 1


if __name__ == '__main__':
    log = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
    best_acc = 0
    start_epoch = 0
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

    trainset = CIFAR10(root='/home/palm/PycharmProjects/DATA/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = CIFAR10(root='/home/palm/PycharmProjects/DATA/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda'
    model = HiResA([3, 3, 3])
    model = torch.nn.DataParallel(model).cuda()

    lambda1 = lambda epoch: epoch // 100
    lambda2 = lambda epoch: 0.95 ** epoch
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-2,
                                momentum=0.9,
                                weight_decay=1e-6, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[150, 225])
    cudnn.benchmark = True

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        scheduler.step()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        log['acc'].append(100.*correct/total)
        log['loss'].append(train_loss/(batch_idx+1))

    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        log['val_acc'].append(100.*correct/total)
        log['val_loss'].append(test_loss/(batch_idx+1))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+300):
        train(epoch)
        test(epoch)
    with open('log/try_{}.json'.format(try_no), 'w') as wr:
        wr.write(log.__str__())
