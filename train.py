from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from utils import *
from model import embed_net
from loss import TripletLoss_WRT, MMDLoss
import math
from torch.cuda.amp import autocast, GradScaler
import os

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam, 0.0007for adamw')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=96, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.5, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=23, type=int, metavar='t', help='random seed')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pool_dim', default=2048)
parser.add_argument('--decay_step', default=16)
parser.add_argument('--warm_up_epoch', default=8, type=int)
parser.add_argument('--max_epoch', default=100)
parser.add_argument('--rerank', default='no', type=str)
parser.add_argument('--dim', default=2048, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)

dataset = args.dataset
model_path = args.model_path + dataset + '/'
if dataset == 'sysu':
    data_path = "../Datasets/SYSU-MM01/"
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = "../Datasets/RegDB/"
    log_path = args.log_path + 'regdb_log/'
    # visible to thermal
    test_mode = [2, 1]
    # thermal to visible
    # test_mode = [1, 2]

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# suffix = dataset
#
# File_name = log_path + suffix + '.log'
# logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=File_name, filemode='a')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)

suffix = dataset
suffix = suffix + '_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])
end = time.time()

if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, args=args)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set(visible to thermal)
    # query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    # gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    # (thermal to visible)
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(class_num=n_class)
net.to(device)
cudnn.benchmark = True

if dataset == 'sysu':
    model_path = "./baseline/sysu_p4_n6_lr_0.1_seed_23_best.t"
if args.dataset == 'regdb':
    model_path = './baseline/regdb_p4_n6_lr_0.1_seed_23_trial_{}_best.t'.format(args.trial)
if dataset == 'llcm':
    model_path = './baseline/llcm_'
if os.path.isfile(model_path):
    print('==> loading checkpoint {}'.format(model_path))
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'], strict=False)
    print('==> loaded checkpoint {}'
          .format(model_path))

assert os.path.isfile(model_path)

criterion_id = nn.CrossEntropyLoss()
# margin=0.45, gamma=64
criterion_tri = TripletLoss_WRT()
# criterion_tri = OriTripletLoss(margin=0.3)
criterion_align = MMDLoss()

criterion_id.to(device)
criterion_tri.to(device)
criterion_align.to(device)

ignored_params = list(map(id, net.bottleneck.parameters())) \
                 + list(map(id, net.classifier.parameters())) \
                 + list(map(id, net.bottleneck_p1.parameters())) \
                 + list(map(id, net.classifier_p1.parameters())) \
                 + list(map(id, net.bottleneck_p2.parameters())) \
                 + list(map(id, net.classifier_p2.parameters())) \
                 + list(map(id, net.bottleneck_p3.parameters())) \
                 + list(map(id, net.classifier_p3.parameters())) \
                 + list(map(id, net.base34_1.base.layer3.parameters())) \
                 + list(map(id, net.base34_1.base.layer4.parameters())) \
                 + list(map(id, net.base34_2.base.layer3.parameters())) \
                 + list(map(id, net.base34_2.base.layer4.parameters())) \

base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * args.lr},
                       {'params': net.bottleneck.parameters(), 'lr': args.lr},
                       {'params': net.classifier.parameters(), 'lr': args.lr},
                       {'params': net.bottleneck_p1.parameters(), 'lr': args.lr},
                       {'params': net.classifier_p1.parameters(), 'lr': args.lr},
                       {'params': net.bottleneck_p2.parameters(), 'lr': args.lr},
                       {'params': net.classifier_p2.parameters(), 'lr': args.lr},
                       {'params': net.bottleneck_p3.parameters(), 'lr': args.lr},
                       {'params': net.classifier_p3.parameters(), 'lr': args.lr},
                       {'params': net.base34_1.base.layer3.parameters(), 'lr': args.lr * 0.2},
                       {'params': net.base34_1.base.layer4.parameters(), 'lr': args.lr * 0.25},
                       {'params': net.base34_2.base.layer3.parameters(), 'lr': args.lr * 0.2},
                       {'params': net.base34_2.base.layer4.parameters(), 'lr': args.lr * 0.25}],
                      weight_decay=5e-4, momentum=0.9, nesterov=True)

warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epoch if epoch <= args.warm_up_epoch else \
    0.5 * (math.cos((epoch - args.warm_up_epoch) / (args.max_epoch - args.warm_up_epoch) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


def adjustLearningRate(epoch, optimizer):
    lr = args.lr

    if epoch < args.decay_step:
        current_lr = lr
    elif epoch >= args.decay_step and epoch < 25:
        current_lr = lr * 0.5
    elif epoch >= 25 and epoch < 30:
        current_lr = lr * 0.25
    elif epoch >= 30 and epoch < 45:
        current_lr = lr * 0.1
    elif epoch >= 45 and epoch < 50:
        current_lr = lr * 0.03
    elif epoch >= 50 and epoch < 80:
        current_lr = lr * 0.01
    else:
        current_lr = lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * current_lr
    for i in range(1, 9):
        optimizer.param_groups[i]['lr'] = current_lr
    optimizer.param_groups[9]['lr'] = current_lr * 0.2
    optimizer.param_groups[10]['lr'] = current_lr * 0.25
    optimizer.param_groups[11]['lr'] = current_lr * 0.2
    optimizer.param_groups[12]['lr'] = current_lr * 0.25

    return optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer


def train(epoch, optimizer, net, args):
    if epoch <= args.decay_step:
        current_lr = optimizer.param_groups[1]['lr']
        base_lr = current_lr * 0.1
        optimizer.param_groups[0]['lr'] = base_lr
        for i in range(1, 9):
            optimizer.param_groups[i]['lr'] = current_lr
        optimizer.param_groups[9]['lr'] = current_lr * 0.2
        optimizer.param_groups[10]['lr'] = current_lr * 0.25
        optimizer.param_groups[11]['lr'] = current_lr * 0.2
        optimizer.param_groups[12]['lr'] = current_lr * 0.25
    else:
        base_lr, current_lr, optimizer = adjustLearningRate(epoch=epoch, optimizer=optimizer)

    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    pid_loss = AverageMeter()
    align_loss1 = AverageMeter()
    align_loss2 = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        optimizer.zero_grad()

        labels = torch.cat((label1, label2, label1, label2), 0)
        labs = torch.cat((label1, label2), 0)
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        labs = Variable(labs.cuda())
        data_time.update(time.time() - end)

        with autocast():
            feat, out0, pout1, pout2, pout3 = net(input1, input2)

            loss_id = criterion_id(out0, labels)
            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            loss_pid1 = criterion_id(pout1, labs)
            loss_pid2 = criterion_id(pout2, labs)
            loss_pid3 = criterion_id(pout3, labs)
            loss_pid = (loss_pid1 + loss_pid2 + loss_pid3) / 3

            vm, tm, vl, tl = feat.chunk(4, 0)
            loss_align1 = criterion_align(F.normalize(vm, p=2, dim=1), F.normalize(tm, p=2, dim=1))
            loss_align2 = criterion_align(F.normalize(vl, p=2, dim=1), F.normalize(tl, p=2, dim=1))

            loss = loss_id + loss_tri + 1.3 * loss_pid + 0.2 * (loss_align1 + loss_align2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        pid_loss.update(loss_pid.item(), 2 * input1.size(0))
        align_loss1.update(loss_align1.item(), 2 * input1.size(0))
        align_loss2.update(loss_align2.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 100 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} '
                  'iLoss: {id_loss.val:.4f} '
                  'TLoss: {tri_loss.val:.4f} '
                  'PLoss: {pid_loss.val:.4f} '
                  'A1Loss: {align_loss1.val:.4f} '
                  'A2Loss: {align_loss2.val:.4f} '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, pid_loss=pid_loss, align_loss1=align_loss1, align_loss2=align_loss2))


def test(epoch):
    # switch to evaluation mode
    with torch.no_grad():
        net.eval()
        # print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat_pool1 = np.zeros((ngall, args.dim))
        gall_feat_fc1 = np.zeros((ngall, args.dim))
        gall_feat_pool2 = np.zeros((ngall, args.dim))
        gall_feat_fc2 = np.zeros((ngall, args.dim))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat_pool, feat_fc = net(input, input, test_mode[0])
                gall_feat_pool1[ptr:ptr + batch_num, :] = feat_pool[:batch_num].detach().cpu().numpy()
                gall_feat_fc1[ptr:ptr + batch_num, :] = feat_fc[:batch_num].detach().cpu().numpy()
                gall_feat_pool2[ptr:ptr + batch_num, :] = feat_pool[batch_num:batch_num*2].detach().cpu().numpy()
                gall_feat_fc2[ptr:ptr + batch_num, :] = feat_fc[batch_num:batch_num*2].detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        # switch to evaluation
        net.eval()
        print('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        query_feat_pool1 = np.zeros((nquery, args.dim))
        query_feat_fc1 = np.zeros((nquery, args.dim))
        query_feat_pool2 = np.zeros((nquery, args.dim))
        query_feat_fc2 = np.zeros((nquery, args.dim))
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input1 = Variable(input.to(device))
                input2 = Variable(torch.flip(input, dims=[3]).to(device))
                feat_pool1, feat_fc1 = net(input1, input1, test_mode[1])

                # TTA to reduce the randomness in testing results
                feat_pool2, feat_fc2 = net(input2, input2, test_mode[1])
                feat_pool = feat_pool1 + feat_pool2
                feat_fc = feat_fc1 + feat_fc2

                query_feat_pool1[ptr:ptr + batch_num, :] = feat_pool[:batch_num].detach().cpu().numpy()
                query_feat_fc1[ptr:ptr + batch_num, :] = feat_fc[:batch_num].detach().cpu().numpy()
                query_feat_pool2[ptr:ptr + batch_num, :] = feat_pool[batch_num:batch_num*2].detach().cpu().numpy()
                query_feat_fc2[ptr:ptr + batch_num, :] = feat_fc[batch_num:batch_num*2].detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        start = time.time()
        distmat_pool1 = np.matmul(query_feat_pool1, np.transpose(gall_feat_pool1))
        distmat_pool2 = np.matmul(query_feat_pool2, np.transpose(gall_feat_pool2))
        distmat_fc1 = np.matmul(query_feat_fc1, np.transpose(gall_feat_fc1))
        distmat_fc2 = np.matmul(query_feat_fc2, np.transpose(gall_feat_fc2))
        distmat_pool = distmat_pool1 + distmat_pool2
        distmat_fc = distmat_fc1 + distmat_fc2
        distmat_pf = distmat_pool + distmat_fc

        # if args.rerank == 'r':
        #     print('reranking.....')
        #     distmat = random_walk(query_feat, gall_feat)
        #     distmat_att = random_walk(query_feat_att, gall_feat_att)
        # elif args.rerank == 'k':
        #     print('reranking.....')
        #     distmat = k_reciprocal(query_feat, gall_feat)
        #     distmat_att = k_reciprocal(query_feat_att, gall_feat_att)
        # else:
        #     distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        #     distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

        # evaluation
        if dataset == 'regdb':
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)
            cmc_fc, mAP_fc, mINP_fc = eval_regdb(-distmat_fc, query_label, gall_label)
            cmc_pf, mAP_pf, mINP_pf = eval_regdb(-distmat_pf, query_label, gall_label)
        elif dataset == 'sysu':
            cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)
            cmc_fc, mAP_fc, mINP_fc = eval_sysu(-distmat_fc, query_label, gall_label, query_cam, gall_cam)
            cmc_pf, mAP_pf, mINP_pf = eval_sysu(-distmat_pf, query_label, gall_label, query_cam, gall_cam)
        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc_pool, mAP_pool, mINP_pool, cmc_fc, mAP_fc, mINP_fc, cmc_pf, mAP_pf, mINP_pf

scaler = GradScaler()

best_epoch = 0
# training
print('==> Start Training...')
for epoch in range(start_epoch, args.max_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                              args.num_pos, args.batch_size, epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, drop_last=True,
                                  num_workers=args.workers)

    # optimizer.zero_grad()
    if epoch <= args.decay_step:
        scheduler.step()
    train(epoch, optimizer, net=net, args=args)

    if (epoch == 0) or ((epoch > 0 and epoch < 30) and epoch % 3 == 0) or (
            (epoch >= 30 and epoch < 50) and epoch % 2 == 0) or (epoch >= 50):
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc_pool, mAP_pool, mINP_pool, cmc_fc, mAP_fc, mINP_fc, cmc_pf, mAP_pf, mINP_pf = test(epoch)
        if cmc_fc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_fc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_fc,
                'mAP': mAP_fc,
                'mINP': mINP_fc,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
        print(
            'PF:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pf[0], cmc_pf[4], cmc_pf[9], cmc_pf[19], mAP_pf, mINP_pf))
        print('Best Epoch [{}]'.format(best_epoch))