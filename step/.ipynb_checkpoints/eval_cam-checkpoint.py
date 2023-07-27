
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from clip_utils import category_dict

n_class = 21

def total_confusion_to_class_confusion(data):

    confusion_c = np.zeros((n_class, 2, 2))
    for i in range(n_class):
        confusion_c[i, 0, 0] = data[i, i]
        confusion_c[i, 0, 1] = np.sum(data[i, :]) - data[i, i]
        confusion_c[i, 1, 0] = np.sum(data[:, i]) - data[i, i]
        confusion_c[i, 1, 1] = np.sum(data) - np.sum(data[i, :]) - np.sum(data[:, i]) + data[i, i]

    return confusion_c


def print_iou(iou, dname='voc'):
    iou_dict = {}
    for i in range(len(iou)-1):
        iou_dict[category_dict[dname][i]] = iou[i+1]
    iou_dict['background'] = iou[0]
    print(iou_dict)

    return iou_dict

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    labels = []
    n_images = 0
    
    for i in range(10, 22):
        t = i / 100.0
    
        for i, id in enumerate(dataset.ids):
            n_images += 1
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=t)#args.cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(dataset.get_example_by_keys(i, (1,))[0])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        # print(iou)
        iou_dict = print_iou(iou, 'voc')

        print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
        print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))

        hyper = [float(h) for h in args.hyper.split(',')]
        name = args.clims_weights_name + f'_{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]}_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth'
        with open(args.work_space + '/eval_result.txt', 'a') as file:
            file.write(name + f' {args.clip} th: {args.cam_eval_thres}, mIoU: {np.nanmean(iou)} {iou_dict} \n')

    return np.nanmean(iou)

def run_V2(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    miou_best,thresh_best = 0,0
    iou_dict_best = None
    preds = []
    labels = []
    n_images = 0
    #th_list = np.arange(0.10,0.22,0.01)
    #for thresh in th_list:
    #    thresh = round(thresh,2)
    #    n_images = 0
    for i, id in enumerate(dataset.ids):
        n_images += 1
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    
   # w-ood  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    confusion_np = np.array(confusion)
    confusion_c = total_confusion_to_class_confusion(confusion_np).astype(float)
    precision, recall = [], []
    for i in range(n_class):
        recall.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, 0, :]))
        precision.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, :, 0]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    # print(iou)
    iou_dict = print_iou(iou, 'voc')
    
    '''
    if np.nanmean(iou) > miou_best:
            miou_best = np.nanmean(iou)
            thresh_best = thresh
            iou_dict_best  = iou_dict
    '''
    
    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
    #print(iou)
    print('precision:',precision)
    print('recall:',recall,'\n\n')
    
    
    '''
    hyper = [float(h) for h in args.hyper.split(',')]
    name = args.clims_weights_name + f'_{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]}_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth'
    with open(args.work_space + '/eval_result.txt', 'a') as file:
        file.write(name + f' {args.clip} th: {thresh_best}, mIoU: {miou_best} {iou_dict_best} \n')
    
    print("[Best]  threshold:", thresh_best, 'miou:', miou_best, "i_imgs", n_images)
    print('iou_dict_best:',iou_dict_best)
    '''
    return np.nanmean(iou),iou_dict

'''
def wood_run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    print('eval to ',args.cam_out_dir)
    preds = []
    labels = []
    n_images = 0
    #th_list = np.arange(0.10,0.22,0.01)
    for i, id in enumerate(dataset.ids):
        n_images += 1
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']

        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    confusion_np = np.array(confusion)
    confusion_c = total_confusion_to_class_confusion(confusion_np).astype(float)
    precision, recall = [], []
    for i in range(n_class):
        recall.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, 0, :]))
        precision.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, :, 0]))

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print("\n")

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images, "precision", np.mean(np.array(precision)), "recall", np.mean(np.array(recall)))
    print("\n")

    iou_dict = print_iou(iou, 'voc')
    #print(iou_dict)
    #print(iou)
    #print(precision)
    #print(recall)
    return np.nanmean(iou), np.mean(np.array(precision)), np.mean(np.array(recall))
'''