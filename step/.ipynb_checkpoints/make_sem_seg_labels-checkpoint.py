import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import voc12.dataloader
from misc import torchutils, indexing
from PIL import Image

cudnn.enabled = True
palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from icecream import ic

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3) #1,3
        d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)
        #d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image, compat=4)
        
        
        Q = d.inference(10)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def _work_crf(process_id, model, dataset, args):
    from misc import torchutils, imutils
    print('CRF version')
    
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )
    
    
    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            #print(img_name)
            # if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.png')):
            #     continue
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            # cams = np.power(cam_dict['cam'], 1.5) # AdvCAM
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            
            
            
            '''
            #ic(rw_up_bg.shape)
            #ic(rw_pred.shape)
            # CRF ~~~~~~~~~~~~~~~~
            img = pack['img'][0].numpy()
            #ic(img[0].shape)
            rw_pred = postprocessor(img[0], rw_up_bg.cpu().numpy())
            '''
            # ~~~~~~~~~~~~~~~~~~~~
            
            ## CRF ~~~~~~~~~~~~~~
            img = pack['img'][0].numpy()
            #ic(all(np.unique(img[0])==np.unique(img[1])))
            #ic(np.unique(img[1]))
            #ic(img.shape)
            #ic(rw_pred.shape)
            rw_pred = imutils.crf_inference_label(np.asarray(img[0].transpose((1, 2, 0))).astype(np.uint8), rw_pred, n_labels=keys.shape[0],t=1)
            #rw_pred = np.argmax(prob, axis=0)
            # ~~~~~~~~~~~~~~~~~~

            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            
            
            #測試加入CGL
            #~~~~~~~~~~~~~~~~~~~~~~~~~~
            #confidence = np.max(prob, axis=0)
            #que = np.unique(confidence)
            #u = que[int(len(que)*0.5)]
            #rw_pred[confidence < u] = 255
            #imageio.imsave(os.path.join('./CGL_test', img_name + '.png'), rw_pred.astype(np.uint8))
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            
            if (iter%1000==0):
                print('{},{}/{}'.format(img_name,iter,len(data_loader)))
            
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            # if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.png')):
            #     continue
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            # cams = np.power(cam_dict['cam'], 1.5) # AdvCAM
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            
            
            if (iter%1000==0):
                print('{},{}/{}'.format(img_name,iter,len(data_loader)))
            
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

def _work_CGL(process_id, model, dataset, args):
    print('CGL')
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            # if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.png')):
            #     continue
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            # cams = np.power(cam_dict['cam'], 1.5) # AdvCAM
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            #print(rw.shape)

            
            
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            prob = rw_up.clone()
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
            
            #print('rw_up:',rw_up.shape)
            #print('rw_up_bg:',rw_up_bg.shape)
            
            rw_pred = keys[rw_pred]
            print("rw_pred",np.unique(rw_pred))
            #CC,_,HH,WW = rw.shape
            
            # CGL
            #rw = rw.view(CC,HH,WW).cpu().numpy()
            #rw_pred =  rw_pred.cpu().numpy()
            confidence = np.max(prob.cpu().numpy(), axis=0)
            #print(confidence.shape)
            que = np.unique(confidence)
            u = que[int(len(que)*0.5)]
            #print(len(que))
            #print(u)
            
            
            #print(confidence.shape)
            #print(rw_pred.shape)
            rw_pred[confidence <0.95] = 255
            print("CGL_rw_pred",np.unique(rw_pred))
            imageio.imsave(os.path.join('./CGL_test', img_name + '.png'), rw_pred.astype(np.uint8))
            #cv2.imwrite(os.path.join(args.pseudo_mask_save_path, image_id + '.png'), label.astype(np.uint8))
            
            #imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            
            
            if (iter%1000==0):
                print('{},{}/{}'.format(img_name,iter,len(data_loader)))
            
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='') 

def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)

    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
    # dataset = voc12.dataloader.VOC12ClassificationDatasetMSF('voc12/train.txt',
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
