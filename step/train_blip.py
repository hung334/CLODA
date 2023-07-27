import cv2
import os
import torch
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib
from imutils import visual_debug
from clip_utils import clip_forward,blip_forward
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss
import voc12.dataloader
from misc import pyutils, torchutils
import os, math

#~~~~~~~~~~~~~~~~~~~~~
# w-ood

import random

from misc import pyutils, torchutils

from step.train_utils import validate, adv_climb, recover_image, KMeans
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
import joblib
from misc import pyutils, torchutils, imutils
from chainercv.datasets import VOCSemanticSegmentationDataset
import numpy as np
from torch.autograd import Variable

from icecream import ic

# Seed
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print('seed:',seed)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#w-ood
def eval_cam_sub(args):
    miou_best = 0
    eval_dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    preds = []
    labels = []
    # n_images = 0
    for thresh in [0.10, 0.12,0.15,0.16,0.18,0.20]:
        for i, id in enumerate(eval_dataset.ids):
            # n_images += 1
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']

            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thresh)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(eval_dataset.get_example_by_keys(i, (1,))[0])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        #confusion_np = np.array(confusion)
        #confusion_c = total_confusion_to_class_confusion(confusion_np).astype(float)


        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        miou = np.nanmean(gtjresj / denominator)
        #print('th:{},miou:{}'.format(thresh,miou))
        if miou > miou_best:
            miou_best = miou

    return miou_best



def sub_cam_eval(args, model):
    print('sub_cam_eval')
    #model_cam = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model_cam = getattr(importlib.import_module(args.clims_network), 'CAM_wood')(n_classes=20)
    
    
    model_cam.load_state_dict(model.module.state_dict(), strict=True)
    model_cam.eval()
    model_cam.cuda()
    dataset_cam = voc12.dataloader.VOC12ClassificationDatasetMSF("voc12/train.txt",
                                                               voc12_root=args.voc12_root, scales=[0.5, 1.0, 1.5, 2.0])
    dataset_cam = torchutils.split_dataset(dataset_cam, 1)

    databin = dataset_cam[0]
    data_loader_cam = DataLoader(databin, shuffle=False, num_workers=2, pin_memory=False)

    with torch.no_grad():
        for iter, pack in enumerate(data_loader_cam):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            # for img in pack['img']:
            #     print(img[0].shape)
            outputs = [model_cam(img[0].cuda(non_blocking=True))
                       for img in pack['img']]  # b x 20 x w x h
            # for o in outputs:
            #     print(o.shape)
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            if len(valid_cat) == 0:
                np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                        {"keys": valid_cat})
                continue
            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            # cv2.imshow('highres', (highres_cam[0].cpu().numpy()*255.0).astype('uint8'))
            # cv2.waitKey(0)
            # save cams

            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
    miou = eval_cam_sub(args)
    return miou




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return

# GLOBAL_SEED = 2
# import numpy as np
# import random
# def set_seed(seed):
#     print('11')
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
# GLOBAL_WORKER_ID = None
# def worker_init_fn(worker_id):
#     global GLOBAL_WORKER_ID
#     GLOBAL_WORKER_ID = worker_id
#     set_seed(GLOBAL_SEED + worker_id)

def run(args):
    
    
    #model = getattr(importlib.import_module(args.clims_network), 'CLIMS')(n_classes=20)
    model = getattr(importlib.import_module(args.clims_network), 'CLIMS_wood')(n_classes=20)
    
    res50_wood = './wood_pth/res50_cam_ood_V3_52231.pth'
    
    if(args.org_cam):
        res50_cam ='wood_pth/res50_cam_orig.pth.pth'#'./wood_pth/res50_cam_ood_V3_52231.pth'
    else:
        res50_cam =res50_wood 
    
    #res50_cam = './wood_pth/res50_cam_ood_V3_52231.pth'
    #'wood_pth/res50_cam_orig.pth.pth'
    #'./wood_pth/res50_cam_ood_V3_52231.pth'
    #'wood_pth/res50_cam_orig.pth.pth'
    #res50_cam_ood_v1_52.pth , res50_cam_orig.pth ,res50_cam_ood_52693
    model.load_state_dict(torch.load(res50_cam), strict=True)
    print('loading ',res50_cam)
    
    
    # initialize backbone network with baseline CAM
    #model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)
    model.cuda()
    #model = torch.nn.DataParallel(model).cuda()
    
    
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.clims_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #w-ood
    train_dataset_ood = voc12.dataloader.OpenImages_ImageDataset(args.ood_list, voc12_root=args.ood_root,
                                                           resize_long=(320, 640), hor_flip=True,
                                                           crop_size=512, crop_method="random", augment=True)

    train_data_loader_ood = DataLoader(train_dataset_ood, batch_size=args.ood_batch_size,
                                       shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    train_data_loader_ood_iter = iter(train_data_loader_ood)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.clims_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.clims_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # Loss
    hyper = [float(h) for h in args.hyper.split(',')]
    OTMLoss = SimMaxLoss()
    BTMLoss = SimMinLoss()
    CBSLoss = BackgroundSuppressionLoss(dname='voc')
    print(hyper)
    

    
    
    # CLIP
    import clip
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip, device=device)
    # for p in clip_model.parameters():
    #     p.requires_grad = False
    clip_model.eval()

    if args.clip == 'RN50x4':
        clip_input_size = 288
    else:
        clip_input_size = 224

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    # BLIP
    
    from BLIP_main.models.blip import blip_feature_extractor
    
    model_url = './BLIP_main/weights/model_base.pth'
    blip_model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='base')
    blip_model.eval()
    blip_model = blip_model.to(device)
    
    
    # transform multi-hot label to class index label
    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()
    
    best_sub_miou = 0
    
    hyper = [float(h) for h in args.hyper.split(',')]
    for ep in range(args.clims_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.clims_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            
            blip = pack['blip']
            blip = blip.cuda()
            
            label = pack['label'].cuda(non_blocking=True)
            '''
            fg_label = preprocess(label.cpu())
            ic(label)
            ic(label.shape)
            ic(fg_label)
            ic(fg_label.shape)
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()
            ic(fg_indices)
            ic(fg_indices.shape)
            '''

            
            fg_label = preprocess(label.cpu())

            x ,cam= model(img)
            N, _, _, _ = x.size()
            optimizer.zero_grad()

            # foreground indices
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()

            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 20, 1, clip_input_size,
                                                                                                clip_input_size)
            img_224 = F.interpolate(blip, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

            fg_224_eval = []
            bg_224_eval = []
            temp_idx = torch.nonzero(label == 1, as_tuple=False)
            for j in range(temp_idx.shape[0]):
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])  #img_224[temp_idx[j, 0]]-第幾張原圖
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])

            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)

            L_OTM = OTMLoss(blip_forward(blip_model, fg_224_eval, fg_label[fg_indices], dname='voc'), 1)
            #print('L_OTM:',L_OTM)
            #print(L_OTM.requires_grad)
            L_BTM = BTMLoss(blip_forward(blip_model, bg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_CBS = torch.mean(x)#CBSLoss(clip_model, fg_224_eval)

            L_REG = torch.mean(x)
            
            
            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG
            loss = hyper[0] * L_OTM + hyper[1] * L_BTM  + hyper[3] * L_REG
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if ep == args.clims_num_epoches - 1:
                if (step > len(train_data_loader) / 4 * 3) and (step % 10 == 0):
                #if True:
                        now_miou = sub_cam_eval(args, model)
                        print('now_miou',now_miou)
                        if now_miou > best_sub_miou:
                            torch.save(model.module.state_dict(), args.clims_weights_name + '_best.pth')
                            best_sub_miou = now_miou
                            print('Step eval_epoch/step:{}/{},best_sub_miou:{}'.format(ep + 1,step,best_sub_miou))
                
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            avg_meter.add({'loss1': loss.item(), 'L_OTM': L_OTM.item(), 'L_BTM': L_BTM.item(), 'L_CBS': L_CBS.item(),
                           'L_REG': L_REG.item(),})

            if (optimizer.global_step - 1) % 200 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'L_OTM:%.4f' % (avg_meter.pop('L_OTM')),
                      'L_BTM:%.4f' % (avg_meter.pop('L_BTM')),
                      'L_CBS:%.4f' % (avg_meter.pop('L_CBS')),
                      'L_REG:%.4f' % (avg_meter.pop('L_REG')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                # visualize class activation maps during training if needed.
                # visual_debug(img, label, x, 'vis/clims_v2_voc12_cam_vis', optimizer.global_step, num_classes=21,
                #             dataset='coco', phase='train')

        # validate(model, val_data_loader)
        now_miou = sub_cam_eval(args, model)

        if now_miou > best_sub_miou:
                     torch.save(model.module.state_dict(), args.clims_weights_name + '_best.pth')
                     best_sub_miou = now_miou
                     print('Step eval___epoch:{},best_sub_miou:{}'.format(ep + 1,best_sub_miou))
        
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.clims_weights_name + '.pth')
    torch.cuda.empty_cache()
