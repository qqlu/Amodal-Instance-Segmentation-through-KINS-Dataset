# [Amodal Instance Segmentation through KINS Dataset](http://jiaya.me/papers/amodel_cvpr19.pdf)
by [Lu Qi](http://www.luqi.info), Li Jiang, [Shu Liu](http://www.shuliu.me), [Xiaoyong Shen](http://xiaoyongshen.me/), [Jiaya Jia](http://www.cse.cuhk.edu.hk/leojia/).

# Update! (16.02.2020)
- **We update the new annotation with occlusion order in update_train_2020.json and update_test_2020.json. The visualization code is in vis_json_2020.py. Please download in here(https://drive.google.com/drive/folders/1FuXz1Rrv5rrGG4n7KcQHVWKvSyr3Tkyo?usp=sharing).** 
- "a_\*" and "i_\*" represent the amodal and inmodal annotation. 
- **The "oco_id" and "ico_id" represent the cluster id and the relative occlusion id in this cluster for instances.** As paper said, instances in an image are first partitioned into several disconnected clusters, each with a few connected instances for easy occlusion detection. Relative occlusion order is based on the distance of each instance to the camera. The non-overlapping instances are labeled as 0. As for the occluded instances in a cluster, order starts from 1 and increases by 1 when occluded once.

## Introduction
This repository has released the training and test set of KINS. The annotation format follows COCO style. The mask can be decoded by COCOAPI.

And the reference code of the method in CVPR 2019 paper '[Amodal Instance Segmentation through KINS Dataset](http://jiaya.me/papers/amodel_cvpr19.pdf)' has been released. The codebase is based on [pytorch-detectron](https://github.com/roytseng-tw/Detectron.pytorch).
You can see some details from our released code. I am sorry that I can not transform it into the maskrcnn-benchmark with clear version.

The dataset could be downloaded from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d. Please download left color images of object data set (http://www.cvlibs.net/download.php?file=data_object_image_2.zip).

## Citation

If our method and dataset are useful for your research, please consider citing:

    @inproceedings{qi2019amodal,
      title={Amodal Instance Segmentation With KINS Dataset},
      author={Qi, Lu and Jiang, Li and Liu, Shu and Shen, Xiaoyong and Jia, Jiaya},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={3014--3023},
      year={2019}
    }


### Contact

Please send email to qqlu1992@gmail.com.
