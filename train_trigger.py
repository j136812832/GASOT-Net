#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !/usr/bin/env python
# coding: utf-8
# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import time
import random
import argparse
import torch
import numpy as np
from torchvision import models
# from model.baseline import Net as Net2
# from model.convnext_base import Net as Net2
# from model.resnet_base import Net as Net2
# from model.convnext_base import Net as Net2
# from model.bisenetV2 import BiSeNetV2 as Net2
# from model.ab_se_de import Net as Net2
# from model.wmyNet import Net as Net2
# from model.wmyNet_gasot import Net as Net2
# from compare_model.unet import UNet as Net2
# from compare_model.segformer_model import Net as Net2
# from compare_model.deeplabv3 import Net as Net2
# from compare_model.DANet import Net as Net2
# from compare_model.segformer_neo import SegFormer as Net2
from compare_model.segformer_b5 import Net as Net2
# from compare_model.bisenet import BiSeNet as Net2
# from compare_model.deeplabv3_neo import Net as Net2
# from compare_model.pspnet import PSPNet as Net2
# from compare_model.fcn import FCN as Net2
import torch.optim as optim
from detail_loss import DetailAggregateLoss, DetailAggregateLoss_sobel_x, DetailAggregateLoss_sobel_y
from loader.load_loveda_nob import vaihingenloader
from torch.utils.data import DataLoader
from metrics.metrics_loveda import runningScore, averageMeter
# from metrics.metrics_potsdam import runningScore, averageMeter
import torch.backends.cudnn as cudnn
from utils.modeltools import netParams
from utils.set_logger import get_logger
import utils.utils
from metrics.CFL import errorloss
import matplotlib.pyplot as plt
import warnings
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from ptflops import get_model_complexity_info

warnings.filterwarnings('ignore')


listLoss = []
listepoch = []
listTestLoss = []
listTestepoch = []

# lr  global
listLrEpoch = []
listLrValue = []





def binarycrossentropy(pred, gts):

    nclass = pred.size(1) // 2
    # convert groundtruth
    gts = [ mask_to_onehot(gts[it], num_classes=nclass) for it in range(gts.size(0))]
    gts = torch.stack(gts, dim=0).long()

    loss = 0.
    for idx in range(nclass):
        loss += F.cross_entropy(pred[:, idx:idx + 2], gts[:, idx:idx + 1].squeeze(1))
    return loss


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = []
    for i in range(num_classes):
        true_mask = mask == i
        _mask.append(true_mask)
        # _mask.append(~true_mask)

    return torch.stack(_mask, dim=0).int()



# setup scheduler
def adjust_learning_rate(cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    cur_iter = cur_epoch * perEpoch_iter + curEpoch_iter
    max_iter = max_epoch * perEpoch_iter
    lr = baselr * pow((1 - 1.0 * cur_iter / max_iter), 0.9)

    # plot lr change
    global listLrEpoch
    listLrEpoch.append(cur_epoch)
    global listLrValue
    listLrValue.append(lr)

    plt.clf()
    # plt.plot(color='r')  # 用蓝色线条绘图
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.plot(listLrEpoch, listLrValue)
    picSave = os.path.join(args.picsave, 'lr.jpg')
    plt.savefig(picSave)

    return lr


def test(args, testloader, model, criterion, epoch, logger):
    '''
    args:
        test_loader: loaded for test dataset
        model: model
    return:
        mean IoU, IoU class
    '''
    model.eval()
    tloss = 0.
    # Setup Metrics
    running_Metrics = runningScore(args.num_classes)
    total_batches = len(testloader)
    print("=====> the number of iterations per epoch: ", total_batches)
    with torch.no_grad():
        for iter, batch in enumerate(testloader):
            # start_time = time.time()
            image, label, name = batch
            image = image[:, 0:3, :, :].cuda()
            label = label.cuda()
            output = model(image)
            loss = criterion(output, label)
            tloss += loss.item()
            # inter_time = time.time() - start_time
            output = output.cpu().detach()[0].numpy()
            gt = np.asarray(label[0].cpu().detach().numpy(), dtype=np.uint8)
            # print('gt size {}, output shape {}'.format(gt.shape, output.shape))
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            running_Metrics.update(gt, output)

    print(f"test phase: Epoch [{epoch:d}/{args.max_epochs:d}] loss: {tloss / total_batches:.5f}")
    score, class_iou, class_F1 = running_Metrics.get_scores()

    running_Metrics.reset()

    # plot train loss
    global listTestLoss
    listTestLoss.append(tloss / total_batches)
    global listTestepoch
    listTestepoch.append(epoch)
    if epoch % 1 == 0:
        plt.clf()
        plt.plot(color='r')
        plt.xlabel("epoch")
        plt.ylabel("test_loss")
        plt.plot(listTestepoch, listTestLoss)
        picSave = os.path.join(args.picsave, 'test_loss.jpg')
        plt.savefig(picSave)


    return score, class_iou, class_F1


def train(args, trainloader, model, criterion, boundary_loss, sobelx, sobely, criterion1, optimizer, epoch, logger):
    '''
    args:
        trainloader: loaded for traain dataset
        model: model
        criterion: loss function
        optimizer: optimizer algorithm, such as Adam or SGD
        epoch: epoch_number
    return:
        average loss
    '''
    model.train()
    total_batches = len(trainloader)
    total_loss = 0.
    for iter, batch in enumerate(trainloader, 0):
        lr = adjust_learning_rate(
            cur_epoch=epoch,
            max_epoch=args.max_epochs,
            curEpoch_iter=iter,
            perEpoch_iter=total_batches,
            baselr=args.lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        start_time = time.time()
        images, labels, name = batch
        images = images[:, 0:3, :, :].cuda()
        # labels = labels.type(torch.float).cuda()

        output = model(images)

        # output, detail1, detail2, detail3, aux, gcls= model(images)
        labels = labels.type(torch.long).cuda()
        #
        loss = criterion(output, labels)
        # loss2 = criterion(o2, labels)
        # loss3 = criterion(o3, labels)
        # loss4 = criterion(o4, labels)
        # loss = loss+loss2+loss3+loss4

        # loss_aux = criterion(aux, labels)
        # loss_gcls = binarycrossentropy(gcls, labels)
        # loss2, loss3 = boundary_loss(detail3, labels)
        # loss4, loss5 = boundary_loss(detail2, labels)
        # loss6, loss7 = boundary_loss(detail1, labels)
        #
        # loss_all =   loss + \
        #              0.4*loss_aux +  \
        #              0.6*loss_gcls
        #              # loss2 + loss3 + \
        #              # loss4 + loss5 + \
        #              # loss6 + loss7
        #
        # total_loss = loss.item() + \
        #              0.4*loss_aux.item() +  \
        #              0.6*loss_gcls.item() +  \
        #              loss2.item() + loss3.item() + \
        #              loss4.item() + loss5.item() + \
        #              loss6.item() + loss7.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        interval_time = time.time() - start_time

        if iter + 1 == total_batches:
            fmt_str = '======> epoch [{:d}/{:d}] cur_lr: {:.6f} loss: {:.5f} time: {:.2f}'
            print_str = fmt_str.format(
                epoch,
                args.max_epochs,
                lr,
                total_loss/total_batches,
                interval_time
            )
            print(print_str)
            logger.info(print_str)

    # plot train loss
    global listLoss
    listLoss.append(total_loss/total_batches)
    global listepoch
    listepoch.append(epoch)
    if epoch % 1 == 0:
        plt.clf()
        plt.plot(color='r')
        plt.xlabel("epoch")
        plt.ylabel("train_loss")
        plt.plot(listepoch, listLoss)
        picSave = os.path.join(args.picsave, 'train_loss.jpg')
        plt.savefig(picSave)


def main(args, logger):
    cudnn.enabled = True  # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True  # cudnn auto-tuner to find the best algorithm to use for
    # our hardware
    # Setup random seed
    # cudnn.deterministic = True # ensure consistent results
    # if benchmark = True, deterministic will be False.

    #plot miou, oa, f1 list
    listMiou = []
    listF1 = []
    listOa = []
    listAcc = []
    listEpoche =[]

    seed = random.randint(1, 10000)

    print('======>random seed {}'.format(seed))
    logger.info('======>random seed {}'.format(seed))

    random.seed(seed)  # python random seed
    np.random.seed(seed)  # set numpy random seed

    torch.manual_seed(seed)  # set random seed for cpu
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed) # set random seed for GPU now
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU

    # Setup device
    # device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # setup DatasetLoader
    train_set = vaihingenloader(root=args.root, split='train')
    test_set = vaihingenloader(root=args.root, split='test')

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,  **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    # setup optimization criterion
    # criterion = utils.utils.cross_entropy2d
    criterion = utils.utils.OhemCELoss(0.7, n_min=16*20*20//16)
    criterion1 = errorloss
    # criterion1 = utils.utils.binarycrossentropy
    boundary_loss = DetailAggregateLoss()
    sobelx = DetailAggregateLoss_sobel_x()
    sobely = DetailAggregateLoss_sobel_y()
    # setup model
    print('======> building network')
    logger.info('======> building network')

    model = Net2(7).cuda()

    # # 网络输入仅为一个input
    # macs, params = get_model_complexity_info(model, (3, 512, 512), print_per_layer_stat=False)
    # print(macs)
    # print(params)

    resnet = models.resnet101(pretrained=True)

    #     print(model)
    if torch.cuda.device_count() > 1:
        device_ids = list(map(int, args.gpu.split(',')))
        #     model = FCNRes101().cuda(device_ids[0])
        # model = UNet(n_channels=3, n_classes=6,).cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    print("======> computing network parameters")
    logger.info("======> computing network parameters")

    total_paramters = netParams(model)
    resnet_paramters = netParams(resnet)
    # vgg_paramters = netParams(vgg_model)
    print("the number of parameters: " + str(total_paramters))
    print("the number of resnet_parameters: " + str(resnet_paramters))
    logger.info("the number of parameters: " + str(total_paramters))

    # # setup optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # setup savedir
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    writer = SummaryWriter(log_dir=args.savedir)

    start_epoch = 0
    flag = True

    best_epoch = 0.
    best_overall = 0.
    best_mIoU = 0.
    best_F1 = 0.

    while flag == True:
        for epoch in range(start_epoch, args.max_epochs):


            print('======> Epoch {} starting train.'.format(epoch))
            logger.info('======> Epoch {} starting train.'.format(epoch))

            train(args, train_loader, model, criterion, boundary_loss, sobelx, sobely, criterion1, optimizer, epoch, logger)
            print('======> Epoch {} train finish.'.format(epoch))
            logger.info('======> Epoch {} train finish.'.format(epoch))

            if epoch % 1 == 0 or (epoch + 1) == args.max_epochs:
                print('Now Epoch {}, starting evaluate on Test dataset.'.format(epoch))
                logger.info('Now starting evaluate on Test dataset.')
                print('length of test set:', len(test_loader))
                logger.info('length of test set: {}'.format(len(test_loader)))

                beginTime = time.time()
                if epoch % 1 == 0:
                    score, class_iou, class_F1 = test(args, test_loader, model, criterion, epoch, logger)
                    endTime = time.time()
                    testTime = endTime - beginTime

                    print("test time is %fs  " % (testTime))

                    count = 0
                    listEpoche.append(epoch)
                    for k, v in score.items():
                        count = count + 1

                        print('{}: {:.5f}'.format(k, v))
                        # print(type(k)) # str
                        # print(type(v)) # float64
                        # oa, acc, frewacc, miou, f1
                        if count == 1:
                            listOa.append(v)
                        elif count == 2:
                            listAcc.append(v)
                        elif count == 4:
                            listMiou.append(v)
                        elif count == 5:
                            listF1.append(v)
                            count = 0
                        else:
                            print("====================")


                        logger.info('======>{0:^18} {1:^10}'.format(k, v))

                    # plot 每隔1个epoch画图
                    if epoch % 1 == 0:
                        plt.clf()
                        plt.plot(color='orange')
                        plt.xlabel("epoch")
                        plt.ylabel("miou")
                        plt.plot(listEpoche, listMiou)
                        saveStr = os.path.join(args.picsave, 'miou.jpg')
                        # print(saveStr)
                        plt.savefig(saveStr)

                        plt.clf()
                        plt.plot(color='orange')
                        plt.xlabel("epoch")
                        plt.ylabel("oa")
                        plt.plot(listEpoche, listOa)
                        saveStr = os.path.join(args.picsave, 'oa.jpg')
                        # print(saveStr)
                        plt.savefig(saveStr)

                        plt.clf()
                        plt.plot(color='orange')
                        plt.xlabel("epoch")
                        plt.ylabel("f1")
                        plt.plot(listEpoche, listF1)
                        saveStr = os.path.join(args.picsave,'f1.jpg')
                        # print(saveStr)
                        plt.savefig(saveStr)

                        plt.clf()
                        plt.plot(color='orange')
                        plt.xlabel("epoch")
                        plt.ylabel("acc")
                        plt.plot(listEpoche, listAcc)
                        saveStr = os.path.join(args.picsave, 'acc.jpg')
                        # print(saveStr)
                        plt.savefig(saveStr)



                    print('Now print class iou')
                    for k, v in class_iou.items():
                        print('{}: {:.5f}'.format(k, v))
                        logger.info('======>{0:^18} {1:^10}'.format(k, v))


                    print('Now print class_F1')
                    for k, v in class_F1.items():
                        print('{}: {:.5f}'.format(k, v))
                        logger.info('======>{0:^18} {1:^10}'.format(k, v))

                    writer.add_scalar('test/Mean_IoU', score["Mean IoU : \t"], epoch)
                    if score["Mean IoU : \t"] > best_mIoU:
                        best_mIoU = score["Mean IoU : \t"]
                        # save model in best mIOU
                        model_file_name = args.savedir + '/model_BestmIOU.pth'
                        torch.save(model.state_dict(), model_file_name)
                        # best_epoch = epoch

                    writer.add_scalar('test/Overall_Acc', score["Overall Acc : \t"], epoch)
                    if score["Overall Acc : \t"] > best_overall:
                        best_overall = score["Overall Acc : \t"]
                        # save model in best overall Acc
                        model_file_name = args.savedir + '/model_BestOA.pth'
                        torch.save(model.state_dict(), model_file_name)
                        best_epoch = epoch

                    writer.add_scalar('test/Mean_F1', score["Mean F1 : \t"], epoch)
                    if score["Mean F1 : \t"] > best_F1:
                        best_F1 = score["Mean F1 : \t"]
                        # save model in best mean F1
                        model_file_name = args.savedir + '/model_BestF1.pth'
                        torch.save(model.state_dict(), model_file_name)
                        # best_epoch = epoch

                    print(f"best mean IoU: {best_mIoU}")
                    print(f"best overall : {best_overall}")
                    print(f"best F1: {best_F1}")
                    print(f"best epoch: {best_epoch}")

        #            #save the model
        #            model_file_name = args.savedir +'/model.pth'
        #            state = {"epoch": epoch+1, "model": model.state_dict()}
        #
        #            if (epoch + 1) == args.max_epochs or epoch % 5 == 0:
        #                print('======> Now begining to save model.')
        #                logger.info('======> Now begining to save model.')
        #                torch.save(state, model_file_name)
        #                print('======> Save done.')
        #                logger.info('======> Save done.')
        #
        if (epoch + 1) == args.max_epochs:
            # print('the best pred mIoU: {}'.format(best_pred))
            flag = False
            break


if __name__ == '__main__':

    import timeit
    #/media/data/zyj/loveDA

    #/media/ssd/zyj/P
    #/media/ssd/zyj/data/V1
    start = timeit.default_timer()
    #/home/lab/nas/zhangyijie/jstars_r1/ablation/potsdam/GFEM
    parser = argparse.ArgumentParser(description='Semantic Segmentation...')   # /media/data/zyj/P  /media/ssd/zhangyijie/GRSL2/rs_code/data/V1
    parser.add_argument('--root', default='/media/data/zyj/loveDA', help='data directory')  # /media/ssd2/zyj/data/V1
    parser.add_argument('--picsave', default="/home/jim/nas/zhangyijie/igarss/loveda/BiSeNet", help="directory to save result picture")
    parser.add_argument('--savedir', default="/home/jim/nas/zhangyijie/igarss/loveda/BiSeNet", help="directory to save the model snapshot")
    parser.add_argument('--gpu', type=str, default="6", help="default GPU devices (1)")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epochs', type=int, default=30, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--lr', default= 0.00003, type=float) # 0.00003
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', type=int, default=16, help=" the number of parallel threads")
    parser.add_argument('--run_id', type=str, default="6", help=" the number of run-id")
    # parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--show_interval', default=10, type=int)
    parser.add_argument('--show_val_interval', default=1000, type=int)
    parser.add_argument('--model', default='resnet50', type=str)
    args = parser.parse_args()
    print('BiSeNet-loveda')
    # run_id = '1'
    run_id = args.run_id
    print('Now run_id {}'.format(run_id))
    # 创建训练文件和结果图片保存路径
    args.savedir = os.path.join(args.savedir, str(run_id))
    args.picsave = os.path.join(args.picsave, str(run_id))
    print(args.savedir)
    print(args.picsave)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.picsave):
        os.makedirs(args.picsave)
    logger = get_logger(args.savedir)
    logger = get_logger(args.picsave)

    logger.info('just do it')

    print('Input arguments:')
    logger.info('======>Input arguments:')

    for key, val in vars(args).items():
        print('======>{:16} {}'.format(key, val))
        logger.info('======> {:16} {}'.format(key, val))

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(int(args.gpu.split(',')[0]))
    else:
        torch.cuda.set_device(int(args.gpu))

    main(args, logger)
    end = timeit.default_timer()
    print("training time:", 1.0 * (end - start) / 3600)
    print('model save in {}.'.format(run_id))
    print('BiSeNet-loveda')