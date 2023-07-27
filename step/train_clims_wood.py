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
from clip_utils import clip_forward
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
#~~~~~~~~~~~~~~~~~~~~~

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
        print('th:{},miou:{}'.format(thresh,miou))
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


def get_Mahalanobis_score(model, data, target, sample_mean, precision, num_classes=40, euclidean=False):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''


    data, target = Variable(data, requires_grad=True).cuda(), Variable(target)

    out, out_features = model.module.feature_list(data)

    out_features = [torch.mean(o.view(o.size(0), o.size(1), -1), 2) for o in out_features]

    # compute Mahalanobis score
    gaussian_scores = []
    for layer_index in range(3):
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            if sample_mean[layer_index][i].sum() == 0:
                term_gau = torch.zeros([16]).cuda()
            else:
                zero_f = out_features[layer_index] - batch_sample_mean
                if euclidean:
                    term_gau = (zero_f ** 2).sum(axis=1)
                else:
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            # print("gau", term_gau)
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        gaussian_scores.append(gaussian_score)

    return gaussian_scores, out, out_features



def get_sample_estimator(args, model, epoch=0):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    credit: https://github.com/pokaxpoka/deep_Mahalanobis_detector
    """
    import sklearn.covariance
    train_dataset_noaug = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root)
    print('train_dataset_noaug ok')
    train_data_loader_noaug = DataLoader(train_dataset_noaug, batch_size=1,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    print('train_data_loader_noaug ok')

    train_dataset_ood_noaug = voc12.dataloader.OpenImages_ImageDataset(args.ood_list, voc12_root=args.ood_root, augment=False, resize_long=(500, 500))
    print('train_dataset_ood_noaug ok')
    
    train_data_loader_ood_noaug = DataLoader(train_dataset_ood_noaug, batch_size=1,
                                       shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    print('train_data_loader_ood_noaug ok')
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    num_output = 3
    num_sample_per_class = np.empty(20+args.cluster_K) # fg20 + fake fg 20 (from ood)
    num_sample_per_class.fill(0)

    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(20+args.cluster_K):
            temp_list.append(torch.Tensor([0]))
        list_features.append(temp_list) # 3 * 40
    # in-distribution
    # supervised center calcaulation
    
    print("get_sample_estimator......")
    with torch.no_grad():
        for img_idx, pack in enumerate(train_data_loader_noaug):

            data, target = pack['img'], pack['label']
            # total += data.size(0)
            data = data.cuda()
            #data = Variable(data, volatile=True)
            data = Variable(data)
            output, out_features = model.feature_list(data)
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
            if img_idx == 0:
                x_to_cluster_voc = out_features[2].view(1, -1)
            else:
                x_to_cluster_voc = torch.cat((x_to_cluster_voc, out_features[2].view(1, -1)), 0)

        # out-of-distribution
        # unsupervised center calculation by K-Means clustering

        img_names = []
        for img_idx, pack in enumerate(train_data_loader_ood_noaug):
            # if img_idx > 80:
            #     break
            data = pack['img']
            img_names.append(pack['name'])
            data = data.cuda()
            # print(data.shape)
            #data = Variable(data, volatile=True)
            data = Variable(data)
            output, out_features = model.feature_list(data)
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
            if img_idx == 0:
                x_to_cluster = out_features[2].view(1, -1)
            else:
                x_to_cluster = torch.cat((x_to_cluster, out_features[2].view(1, -1)), 0)

    #print(x_to_cluster.shape, x_to_cluster_voc.shape)



    for img_idx, pack in enumerate(train_data_loader_noaug):
        target = pack['label']
        labels = torch.nonzero(target[0])[:, 0]
        for label in labels:
            if num_sample_per_class[label] == 0:
                list_features[2][label] = x_to_cluster_voc[img_idx].view(1, -1)

            else:
                list_features[2][label] = torch.cat((list_features[2][label], x_to_cluster_voc[img_idx].view(1, -1)))

            num_sample_per_class[label] += 1

    ood_label, ood_center = KMeans(x_to_cluster, K=args.cluster_K, Niter=10)


    for feat_idx, feat in enumerate(x_to_cluster):
        label = ood_label[feat_idx] + 20
        feat = feat.view(1, -1)
        if num_sample_per_class[label] == 0:
            list_features[2][label] = feat
        else:
            list_features[2][label] = torch.cat((list_features[2][label], feat))
        num_sample_per_class[label] += 1


    sample_class_mean = []
    out_count = 0
    for num_feature in [0, 0, 2048]:
        temp_list = torch.Tensor(20+args.cluster_K, int(num_feature)).cuda()
        if num_feature == 0:
            sample_class_mean.append(temp_list)
            out_count += 1
            continue

        for j in range(20+args.cluster_K):
            if len(list_features[out_count][j].shape) == 1:
                print("nope!")
                temp_list[j] = torch.zeros([num_feature]).cuda()
                continue
            else:
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1


    precision = []
    for k in range(num_output):
        X = 0
        if k != 2:
            precision.append(0)
            continue

        for i in range(20+args.cluster_K):
            if len(list_features[k][i].shape) == 1:
                continue
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    return sample_class_mean, precision

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
    
    res50_cam = 'wood_pth/res50_cam_orig.pth.pth'
    #res50_cam_ood_v1_52.pth , res50_cam_orig.pth ,res50_cam_ood_52693
    model.load_state_dict(torch.load(res50_cam), strict=True)
    print('loading ',res50_cam)
    
    
    # initialize backbone network with baseline CAM
    #model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)
    model.cuda()
    #model = torch.nn.DataParallel(model).cuda()
    
    print('get_sample_estimator ing......')
    sample_class_mean, precision = get_sample_estimator(args, model, epoch=0)
    #sample_class_mean = torch.load('sample_class_mean.pt')
    #precision = torch.load('precision.pt')
    print('get_sample_estimator ok')
    print("sample_class_mean:",sample_class_mean)
    #torch.save(sample_class_mean, 'sample_class_mean.pt')
    #torch.save(precision, 'precision.pt')
    print("precision:",precision)
    
    
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
            label = pack['label'].cuda(non_blocking=True)
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # w-ood
            M_distances, output, out_features = get_Mahalanobis_score(model, img, label, sample_class_mean, precision, num_classes=20+args.cluster_K, euclidean=True)
            M_distances[2] = torch.sqrt(M_distances[2])
            fg_distances = M_distances[2][:, :20] * label
            ood_distances = M_distances[2][:, 20:]

            diff = ood_distances
            topk_value, _ = torch.topk(diff[diff.nonzero(as_tuple=True)], k=int(diff[diff.nonzero(as_tuple=True)].shape[0]*(1-args.ood_dist_topk)))

            thresh = topk_value[-1].item()

            loss_M = fg_distances.mean() * (20 / args.cluster_K) - ((diff) * (diff < thresh)).mean()
            
            # bg loss
            if args.ood_coeff > 0:
                try:
                    pack_ood = next(train_data_loader_ood_iter)
                except:
                    train_data_loader_ood_iter = iter(train_data_loader_ood)
                    pack_ood = next(train_data_loader_ood_iter)
                img_ood = pack_ood['img'].cuda()


                label_ood = torch.zeros([args.ood_batch_size, 20]).cuda()

                for adv_iter in range(1):
                    img_ood.requires_grad = True
                    logit_ood, out_features_ood = model.module.feature_list(img_ood)
                    logit_max, _ = torch.max(logit_ood, dim=1)

                    loss_adv = logit_max.sum()
                    model.zero_grad()
                    loss_adv.backward()
                    data_grad = img_ood.grad.data
                    perturbed_data = adv_climb(img_ood, 0.07, data_grad).detach()
                    img_ood = perturbed_data
                logit_ood_perturb, out_features_ood_perturb = model.module.feature_list(perturbed_data)
                logit_max_perturb, logit_max_perturb_idx = torch.max(logit_ood_perturb, dim=1)

                loss_mask = (logit_max_perturb_idx != 4)
                loss_ood = F.multilabel_soft_margin_loss(logit_ood_perturb, label_ood, reduce=False)
                loss_ood = loss_ood * loss_mask

                loss_ood = loss_ood.sum() / args.cam_batch_size
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
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

            '''
            fg_label = preprocess(label.cpu())

            x = model(img)
            N, _, _, _ = x.size()
            optimizer.zero_grad()

            # foreground indices
            fg_indices = torch.nonzero(label.reshape(-1) == 1, as_tuple=False).squeeze()

            cam_224 = F.interpolate(x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * 20, 1, clip_input_size,
                                                                                                clip_input_size)
            img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

            fg_224_eval = []
            bg_224_eval = []
            temp_idx = torch.nonzero(label == 1, as_tuple=False)
            for j in range(temp_idx.shape[0]):
                fg_224_eval.append(cam_224[fg_indices[j]] * img_224[temp_idx[j, 0]])  #img_224[temp_idx[j, 0]]-第幾張原圖
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * img_224[temp_idx[j, 0]])

            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)

            L_OTM = OTMLoss(clip_forward(clip_model, fg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_BTM = BTMLoss(clip_forward(clip_model, bg_224_eval, fg_label[fg_indices], dname='voc'), 1)

            L_CBS = CBSLoss(clip_model, fg_224_eval)

            L_REG = torch.mean(x)
            '''
            
            L_OTM,L_BTM,L_CBS,L_REG = loss_M,loss_M,loss_M,loss_M
            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM + hyper[2] * L_CBS + hyper[3] * L_REG
            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM  + hyper[3] * L_REG
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # w-ood
            L_cross = F.binary_cross_entropy_with_logits(output, label) 
            #L_cross = F.multilabel_soft_margin_loss(output, label) 
            L_wood =  loss_M * args.distance_lambda + loss_ood * args.ood_coeff
            #L_cross,loss_M ,L_wood,loss_ood = L_OTM,L_OTM,L_OTM,L_OTM
            
            #loss = (hyper[0] * L_OTM + hyper[1] * L_BTM  + hyper[3] * L_REG)*0.0025 + L_wood + L_cross
            #loss = hyper[0] * L_OTM + hyper[1] * L_BTM  + hyper[3] * L_REG #+ L_wood #+ L_cross
            loss =  F.binary_cross_entropy_with_logits(output, label) + loss_M * args.distance_lambda + loss_ood * args.ood_coeff#L_wood + L_cross
            
            
            #'10,24,1,0.2_16'
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
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
                           'L_REG': L_REG.item(),
                          'L_cross': L_cross.item(),
                          'L_wood':L_wood.item(),
                          'L_wood_M':loss_M.item(),
                          'L_wood_ood':loss_ood.item(),})

            if (optimizer.global_step - 1) % 200 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'L_OTM:%.4f' % (avg_meter.pop('L_OTM')),
                      'L_BTM:%.4f' % (avg_meter.pop('L_BTM')),
                      'L_CBS:%.4f' % (avg_meter.pop('L_CBS')),
                      'L_REG:%.4f' % (avg_meter.pop('L_REG')),
                      'L_cross:%.4f' % (avg_meter.pop('L_cross')),
                      'L_wood:%.4f' % (avg_meter.pop('L_wood')),
                      'L_wood_M:%.4f' % (avg_meter.pop('L_wood_M')),
                      'L_wood_ood:%.4f' % (avg_meter.pop('L_wood_ood')),
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