# YOLOv4

This is PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet) which is based on [ultralytics/yolov3](https://github.com/ultralytics/yolov3).

* [[original Darknet implementation of YOLOv4]](https://github.com/AlexeyAB/darknet)

* [[ultralytics/yolov5 based PyTorch implementation of YOLOv4]](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5).

### development log

<details><summary> <b>Expand</b> </summary>

* `2021-01-26` - design mask objectness.
* `2021-01-25` - design rotate augmentation.
* `2021-01-23` - design collage augmentation.
* `2021-01-22` - support [VoVNet](https://arxiv.org/abs/1904.09730), [VoVNetv2](https://arxiv.org/abs/1911.06667).
* `2021-01-22` - support [EIoU](https://arxiv.org/abs/2101.08158).
* `2021-01-19` - support instance segmentation. [`mask-yolo`]()
* `2021-01-17` - support anchor-free-based methods. [`center-yolo`]()
* `2021-01-14` - support joint detection and classification. [`classify-yolo`]()
* `2020-01-02` - design new [PRN](https://github.com/WongKinYiu/PartialResidualNetworks) and [CSP](https://github.com/WongKinYiu/CrossStagePartialNetworks)-based models.
* `2020-12-22` - support transfer learning.
* `2020-12-18` - support non-local series self-attention blocks. [`gc`](https://arxiv.org/abs/1904.11492) [`dnl`](https://arxiv.org/abs/2006.06668)
* `2020-12-16` - support down-sampling blocks in cspnet paper. [`down-c`]() [`down-d`](https://arxiv.org/abs/1812.01187)
* `2020-12-03` - support imitation learning.
* `2020-12-02` - support [squeeze and excitation](https://arxiv.org/abs/1709.01507).
* `2020-11-26` - support multi-class multi-anchor joint detection and embedding.
* `2020-11-25` - support [joint detection and embedding](https://arxiv.org/abs/1909.12605). [`track-yolo`]()
* `2020-11-23` - support teacher-student learning.
* `2020-11-17` - pytorch 1.7 compatibility. 
* `2020-11-06` - support inference with initial weights. 
* `2020-10-21` - fully supported by darknet. 
* `2020-09-18` - design fine-tune methods. 
* `2020-08-29` - support [deformable kernel](https://arxiv.org/abs/1910.02940).
* `2020-08-25` - pytorch 1.6 compatibility.
* `2020-08-24` - support channel last training/testing. 
* `2020-08-16` - design CSPPRN. 
* `2020-08-15` - design deeper model. [`csp-p6-mish`]()
* `2020-08-11` - support [HarDNet](https://arxiv.org/abs/1909.00948). [`hard39-pacsp`]() [`hard68-pacsp`]() [`hard85-pacsp`]()
* `2020-08-10` - add DDP training.
* `2020-08-06` - support [DCN](https://arxiv.org/abs/1703.06211), [DCNv2](https://arxiv.org/abs/1811.11168). [`yolov4-dcn`]()
* `2020-08-01` - add pytorch hub.
* `2020-07-31` - support [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [CSPResNet](https://github.com/WongKinYiu/CrossStagePartialNetworks), [CSPResNeXt](https://github.com/WongKinYiu/CrossStagePartialNetworks). [`r50-pacsp`]() [`x50-pacsp`]() [`cspr50-pacsp`]() [`cspx50-pacsp`]()
* `2020-07-28` - support [SAM](https://arxiv.org/abs/2004.10934). [`yolov4-pacsp-sam`]()
* `2020-07-24` - update api.
* `2020-07-23` - support CUDA accelerated Mish activation function.
* `2020-07-19` - support and training tiny YOLOv4. [`yolov4-tiny`]()
* `2020-07-15` - design and training conditional YOLOv4. [`yolov4-pacsp-conditional`]()
* `2020-07-13` - support [MixUp](https://arxiv.org/abs/1710.09412) data augmentation.
* `2020-07-03` - design new stem layers.
* `2020-06-16` - support floating16 of GPU inference.
* `2020-06-14` - convert .pt to .weights for darknet fine-tuning.
* `2020-06-13` - update multi-scale training strategy.
* `2020-06-12` - design scaled YOLOv4 follow [ultralytics](https://github.com/ultralytics/yolov5). [`yolov4-pacsp-s`]() [`yolov4-pacsp-m`]() [`yolov4-pacsp-l`]() [`yolov4-pacsp-x`]()
* `2020-06-07` - design [scaling methods](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/images/scalingCSP.png) for CSP-based models. [`yolov4-pacsp-25`]() [`yolov4-pacsp-75`]()
* `2020-06-03` - update COCO2014 to COCO2017.
* `2020-05-30` - update FPN neck to CSPFPN. [`yolov4-yocsp`]() [`yolov4-yocsp-mish`]()
* `2020-05-24` - update neck of YOLOv4 to CSPPAN. [`yolov4-pacsp`]() [`yolov4-pacsp-mish`]()
* `2020-05-15` - training YOLOv4 with Mish activation function. [`yolov4-yospp-mish`]() [`yolov4-paspp-mish`]()
* `2020-05-08` - design and training YOLOv4 with [FPN](https://arxiv.org/abs/1612.03144) neck. [`yolov4-yospp`]()
* `2020-05-01` - training YOLOv4 with Leaky activation function using PyTorch. [`yolov4-paspp`]() [`PAN`](https://arxiv.org/abs/1803.01534)

</details>

## Pretrained Models & Comparison

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4** | 672 | 47.7% | 66.7% | 52.1% | 30.5% | 52.6% | 61.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4.cfg) | [weights](https://drive.google.com/file/d/137U-oLekAu-J-fe0E_seTblVxnU3tlNC/view?usp=sharing) |
|  |  |  |  |  |  |  |
| **YOLOv4**<sub>pacsp-s</sub> | 672 | 36.6% | 55.5% | 39.6% | 21.2% | 41.1% | 47.0% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-s.cfg) | [weights](https://drive.google.com/file/d/1-QZc043NMNa_O0oLaB3r0XYKFRSktfsd/view?usp=sharing) |
| **YOLOv4**<sub>pacsp</sub> | 672 | 47.2% | 66.2% | 51.6% | 30.4% | 52.3% | 60.8% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp.cfg) | [weights](https://drive.google.com/file/d/1sIpu29jEBZ3VI_1uy2Q1f3iEzvIpBZbP/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-x</sub> | 672 | **49.3%** | **68.1%** | **53.6%** | **31.8%** | **54.5%** | **63.6%** | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-x.cfg) | [weights](https://drive.google.com/file/d/1aZRfA2CD9SdIwmscbyp6rXZjGysDvaYv/view?usp=sharing) |
|  |  |  |  |  |  |  |
| **YOLOv4**<sub>pacsp-s-mish</sub> | 672 | 38.6% | 57.7% | 41.8% | 22.3% | 43.5% | 49.3% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-s-mish.cfg) | [weights](https://drive.google.com/file/d/1q0zbQKcSNSf_AxWQv6DAUPXeaTywPqVB/view?usp=sharing) |
| (+BoF) | 640 | 39.9% | 59.1% | 43.1% | 24.4% | 45.2% | 51.4% |  | [weights](https://drive.google.com/file/d/1-8PqBaI8oYb7TB9L-KMzvjZcK_VaGXCF/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-mish</sub> | 672 | 48.1% | 66.9% | 52.3% | 30.8% | 53.4% | 61.7% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-mish.cfg) | [weights](https://drive.google.com/file/d/116yreAUTK_dTJErDuDVX2WTIBcd5YPSI/view?usp=sharing) |
| (+BoF) | 640 | 49.3% | 68.2% | 53.8% | 31.9% | 54.9% | 62.8% |  | [weights](https://drive.google.com/file/d/12qRrqDRlUElsR_TI97j4qkrttrNKKG3k/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-x-mish</sub> | 672 | 50.0% | 68.5% | 54.4% | 32.9% | 54.9% | 64.0% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-x-mish.cfg) | [weights](https://drive.google.com/file/d/1GGCrokkRZ06CZ5MUCVokbX1FF2e1DbPF/view?usp=sharing) |
| (+BoF) | 640 | **51.0%** | **69.7%** | **55.5%** | **33.3%** | **56.2%** | **65.5%** |  | [weights](https://drive.google.com/file/d/1lVmSqItSKywg6yk1qiCvgOYw55O03Qgj/view?usp=sharing) |
|  |  |  |  |  |  |  |

## Requirements

```
pip install -r requirements.txt
```
※ For running Mish models, please install https://github.com/thomasbrandon/mish-cuda

## Training

```
python train.py --device 0 --batch-size 16 --img 640 640 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name yolov4-pacsp
```

## Testing

```
python test.py --img 640 --conf 0.001 --batch 8 --device 0 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights weights/yolov4-pacsp.pt
```

## Citation

```
@article{bochkovskiy2020yolov4,
  title={{YOLOv4}: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```

```
@inproceedings{wang2020cspnet,
  title={{CSPNet}: A New Backbone That Can Enhance Learning Capability of {CNN}},
  author={Wang, Chien-Yao and Mark Liao, Hong-Yuan and Wu, Yueh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={390--391},
  year={2020}
}
```

## Acknowledgements

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
