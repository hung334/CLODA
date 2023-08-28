# CLODA: Cross Language Image Matching Based on Out-of-Distribution Data and Convolutional Block Attention Module for Weakly Supervised Semantic Segmentation

### 架構圖
![image](https://github.com/hung334/CLODA/blob/master/CLODA.png)


## 1.準備數據集
### PASCAL VOC2012
You will need to download the images (JPEG format) in PASCAL VOC2012 dataset at [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and train_aug ground-truth can be found at [here](http://home.bharathh.info/pubs/codes/SBD/download.html). Make sure your `data/VOC2012 folder` is structured as follows:

```
├── VOC2012/
|   ├── Annotations
|   ├── ImageSets
|   ├── SegmentationClass
|   ├── SegmentationClassAug
|   └── SegmentationObject
```

## 2.訓練W-OoD
cd `./CLODA/w-ood-main/`

準備預訓練權重:

- Pre-trained model used in this paper: [Download](https://drive.google.com/file/d/1Eaa7BV6PAfRPEZYBz5WtllUJxpnO-a-m/view?usp=sharing).
- Move it to "sess/"

準備分佈外數據:

- Download OoD images: [Download](https://drive.google.com/file/d/1Zrwqiy-dt9aymtEzCt9qqWROMDj3EUUX)
- Unzip and move it to "WOoD_dataset/openimages"
- Download a list of OoD images [here](https://drive.google.com/file/d/1KjK55YL1jGHgA0LVA3djHu6vtFyho8kP/view?usp=share_link) and move 'ood_list.txt' to "WOoD_dataset/openimages"

訓練指令:

```python
CUDA_VISIBLE_DEVICES=0 python run_sample.py --work_space="result/train_wood" --train_cam_pass True --make_cam_pass True --eval_cam_pass True  --cam_learning_rate 0.01 --voc12_root='../VOCdevkit/VOC2012/'
```

## 3.訓練CLODA

cd `./CLODA-master`

將訓練好的wood權重放入`./wood_pth`

訓練指令:

```python
CUDA_VISIBLE_DEVICES=0 python3 run_sample.py --voc12_root ../VOCdevkit/VOC2012/ --hyper 10,24,1,0.2 --clims_num_epoches 15 --cam_eval_thres 0.15 --work_space ./result/CLODA --cam_network net.resnet50_clims --train_clims_wood_clims_idea2_pass True --make_clims_pass True --eval_cam_pass True --clims_all True --z=0.1 
```

細化+生成偽標籤 指令:

```python
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ../VOCdevkit/VOC2012/ --cam_eval_thres 0.15 --work_space ./result/CLODA --cam_network net.resnet50_clims --make_clims_pass True --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True --clims_weights_name='res50_clims_best' --infer_list voc12/train_aug.txt --num_workers=1
```

## 4.偽標籤訓練DeepLabv2

    
將偽標籤放置好路徑`../sem_seg/sem_seg_7533`

更改指向config 路徑 :
    
  ```python
    
  --config-path configs/voc12_imagenet_pretrained_7533.yaml
  --config-path configs/voc12_coco_pretrained_7533.yaml
  ```
    
  運行:
    
  ```python
  CUDA_VISIBLE_DEVICES=0 bash run_voc12_coco_pretrained.sh
  CUDA_VISIBLE_DEVICES=0 bash run_voc12_imagenet_pretrained.sh
  ```

# Demo
https://youtu.be/3Iz0N3nJvZ4
https://youtu.be/xOGu4LfHNd8
