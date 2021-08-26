
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net_CAM')()

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            img_ref = pack['img_ref']
            label_ref = pack['label_ref'].cuda(non_blocking=True)

            common_label = pack['common_cls']

            x,y,x_cam,y_cam,cam_x,cam_y = model(img,img_ref)

            N, C, H, W = x_cam.size()
            # print(x.shape,y.shape,x_cam.shape,y_cam.shape,cam_x.shape,cam_y.shape)

            loss1 = F.multilabel_soft_margin_loss(x, label)
            loss2 = F.multilabel_soft_margin_loss(y, label_ref)

            label = label.unsqueeze(2).unsqueeze(3)
            label_ref = label_ref.unsqueeze(2).unsqueeze(3)

            cam_sn = F.relu(x_cam)
            cam_sn_max = torch.max(cam_sn.view(N,C,-1), dim=-1)[0].view(N,C,1,1)+1e-5
            cam_sn = F.relu(cam_sn-1e-5, inplace=True)/cam_sn_max
            cam_sn_x = cam_sn * label

            cam_rn = F.relu(cam_x)
            cam_rn_max = torch.max(cam_rn.view(N,C,-1), dim=-1)[0].view(N,C,1,1)+1e-5
            cam_rn = F.relu(cam_rn-1e-5, inplace=True)/cam_rn_max
            cam_rn_x = cam_rn * label

            cam_sn = F.relu(y_cam)
            cam_sn_max = torch.max(cam_sn.view(N,C,-1), dim=-1)[0].view(N,C,1,1)+1e-5
            cam_sn = F.relu(cam_sn-1e-5, inplace=True)/cam_sn_max
            cam_sn_y = cam_sn * label_ref

            cam_rn = F.relu(cam_y)
            cam_rn_max = torch.max(cam_rn.view(N,C,-1), dim=-1)[0].view(N,C,1,1)+1e-5
            cam_rn = F.relu(cam_rn-1e-5, inplace=True)/cam_rn_max
            cam_rn_y = cam_rn * label_ref

            loss_x = torch.mean(torch.pow(cam_sn_x - cam_rn_x, 2)) * 10
            loss_y = torch.mean(torch.pow(cam_sn_y - cam_rn_y, 2)) * 10

            loss = loss1 + loss2 + loss_x + loss_y

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'loss2-4 is', loss2.item(),loss_x.item(),loss_y.item(),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        #else:
        #    validate(model, val_data_loader)
        #    timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()
