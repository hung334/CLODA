import argparse
import os
import numpy as np
import os.path as osp

from misc import pyutils
import random
import torch

def seed_torch(seed=1):
     print('seed:',seed)
     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False

if __name__ == '__main__':

    seed_torch(seed=1)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # w-ood 參數
    # Ood Config
    parser.add_argument("--ood_root", default='../w-ood-main/WOoD_dataset/openimages/OoD_images', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--ood_list", default='../w-ood-main/WOoD_dataset/openimages/ood_list.txt', type=str)
    parser.add_argument("--ood_coeff", default=0.25, type=float)
    parser.add_argument("--ood_batch_size", default=16, type=int)
    parser.add_argument("--cluster_K", default=50, type=int)
    parser.add_argument("--distance_lambda", default=0.007, type=float)
    parser.add_argument("--ood_dist_topk", default=0.2, type=float)
    
    
    parser.add_argument("--m", default=1, type=float)
    parser.add_argument("--z", default=0.2, type=float)
    parser.add_argument("--s", default=0.1, type=float)
    parser.add_argument("--clims_all", type=str2bool, default=False)
    parser.add_argument("--org_cam", type=str2bool, default=False)
    parser.add_argument("--eval_trainaug", type=str2bool, default=False)
    parser.add_argument("--add_cam", type=str2bool, default=False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--voc12_root", default='/data1/xjheng/dataset/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    parser.add_argument("--num_cores_eval", default=8, type=int)

    # CLIMS
    parser.add_argument("--clims_network", default="net.resnet50_clims", type=str)
    parser.add_argument("--clims_num_epoches", default=15, type=int)
    parser.add_argument("--clims_learning_rate", default=0.00025, type=float)
    parser.add_argument('--hyper', default='10,24,1,0.2', type=str)
    parser.add_argument('--clip', default='ViT-B/32', type=str)


    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.3, type=float)
    parser.add_argument("--conf_bg_thres", default=0.1, type=float)


    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.2)

    # Output Path
    parser.add_argument("--work_space", default="result_default5", type=str) # set your path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="ins_seg", type=str)
    parser.add_argument("--clims_weights_name", default="res50_clims", type=str)

    # Step
    parser.add_argument("--train_clims_wood_pass", type=str2bool, default=False)
    parser.add_argument("--train_wood_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_wood_clims_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_wood_clims_idea1_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_wood_clims_idea2_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_wood_clims_idea3_pass", type=str2bool, default=False)
    parser.add_argument("--train_blip_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_CBAM_v2_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_AMM_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_GCNet_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_wood_clims_idea2_res101_pass", type=str2bool, default=False)
    
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--train_clims_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_clims_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=False)
    parser.add_argument("--train_irn_pass", type=str2bool, default=False)
    parser.add_argument("--make_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--make_sem_seg_pass", type=str2bool, default=False) 
    parser.add_argument("--eval_sem_seg_pass", type=str2bool, default=False)

    args = parser.parse_args()
    args.log_name = osp.join(args.work_space,args.log_name)
    args.cam_weights_name = osp.join(args.work_space,args.cam_weights_name)
    args.irn_weights_name = osp.join(args.work_space,args.irn_weights_name)
    args.cam_out_dir = osp.join(args.work_space,args.cam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space,args.ir_label_out_dir)
    args.sem_seg_out_dir = osp.join(args.work_space,args.sem_seg_out_dir)
    args.ins_seg_out_dir = osp.join(args.work_space,args.ins_seg_out_dir)
    args.clims_weights_name = osp.join(args.work_space, args.clims_weights_name)
    
    
    os.makedirs(osp.join(args.work_space,'look_sal'), exist_ok=True)
    
    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    
    if args.train_clims_GCNet_pass is True:
        import step.train_clims_GCNet

        timer = pyutils.Timer('step.train_clims_GCNet:')
        step.train_clims_GCNet.run(args)  
    
    if args.train_clims_AMM_pass is True:
        import step.train_clims_amm

        timer = pyutils.Timer('step.train_clims_amm:')
        step.train_clims_amm.run(args)   
    
    
    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.train_wood_pass is True:
        
        '''
        import step.train_wood_clims

        timer = pyutils.Timer('step.train_wood_clims:')
        step.train_wood_clims.run(args)
        '''
        import step.train_cam_clustering

        timer = pyutils.Timer('step.train_cam_clustering:')
        step.train_cam_clustering.run(args)

        
    if args.train_blip_pass is True:
        
        import step.train_blip

        timer = pyutils.Timer('step.train_blip:')
        step.train_blip.run(args) 
        

    if args.train_clims_wood_pass is True:
        
        import step.train_clims_wood

        timer = pyutils.Timer('step.train_clims_wood:')
        step.train_clims_wood.run(args) 
        
    if args.train_clims_wood_clims_idea1_pass is True:
        
        import step.train_clims_wood_clims_idea1

        timer = pyutils.Timer('step.train_clims_wood_clims_idea1:')
        step.train_clims_wood_clims_idea1.run(args)

    if args.train_clims_CBAM_v2_pass is True:
        
        import step.train_clims_CBAM_v2

        timer = pyutils.Timer('step.train_clims_CBAM_v2:')
        step.train_clims_CBAM_v2.run(args)        
    
    
    if args.train_clims_wood_clims_idea2_res101_pass is True:
        
        import step.train_clims_wood_clims_idea2_res101

        timer = pyutils.Timer('step.train_clims_wood_clims_idea2_res101:')
        step.train_clims_wood_clims_idea2_res101.run(args)
    
    if args.train_clims_wood_clims_idea2_pass is True:
        
        import step.train_clims_wood_clims_idea2

        timer = pyutils.Timer('step.train_clims_wood_clims_idea2:')
        step.train_clims_wood_clims_idea2.run(args)
    
    if args.train_clims_wood_clims_idea3_pass is True:
        
        import step.train_clims_wood_clims_idea3

        timer = pyutils.Timer('step.train_clims_wood_clims_idea3:')
        step.train_clims_wood_clims_idea3.run(args)
        
    if args.train_clims_wood_clims_pass is True:
        
        import step.train_clims_wood_clims

        timer = pyutils.Timer('step.train_clims_wood_clims:')
        step.train_clims_wood_clims.run(args)
    
    
    if args.train_clims_pass is True:
        
        import step.train_clims

        timer = pyutils.Timer('step.train_clims:')
        step.train_clims.run(args)
        

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)
    
    if args.make_clims_pass is True:
        import step.make_clims

        timer = pyutils.Timer('step.make_clims:')
        step.make_clims.run(args)

    if args.eval_cam_pass is True:
        
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        
        miou_best,thresh_best = 0,0
        iou_dict_best = None
        th_list = np.arange(0.10,0.19,0.01)
        for thresh in th_list:
            args.cam_eval_thres = round(thresh,2)
            miou, iou_dict = step.eval_cam.run_V2(args)
            if(miou>miou_best):
                miou_best = miou
                thresh_best = args.cam_eval_thres
                iou_dict_best = iou_dict
                
        hyper = [float(h) for h in args.hyper.split(',')]
        name = args.clims_weights_name +f'_{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]}_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth'
        with open(args.work_space + '/eval_result.txt', 'a') as file:
            file.write(name + f' {args.clip} th: {thresh_best}, mIoU: {miou_best} {iou_dict_best} \n')

        print("[Best]  threshold:", thresh_best, 'miou:', miou_best)
        print('iou_dict_best:',iou_dict_best)
        
        '''
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run_V2(args)
        '''
        
    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        
        if args.eval_trainaug :
            step.eval_sem_seg.run_train_aug(args)
            
        else:
        
            step.eval_sem_seg.run(args)

