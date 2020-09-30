# YOLOv4

This is PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet) which is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

* [[original Darknet implementation of YOLOv4]](https://github.com/AlexeyAB/darknet)

* [[ultralytics/yolov3 based PyTorch implementation of YOLOv4]](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview).

### development log

<details><summary> <b>Expand</b> </summary>

* `2020-08-29` - support deformable kernel. 
* `2020-08-24` - support channel last training/testing. 
* `2020-08-16` - design CSPPRN. 
* `2020-08-15` - design deeper model. [`csp-p6-mish`]()
* `2020-08-11` - support HarDNet. [`hard39-pacsp`]() [`hard68-pacsp`]() [`hard85-pacsp`]()
* `2020-08-10` - add DDP training.
* `2020-08-06` - support DCN, DCNv2. [`yolov4-dcn`]()
* `2020-08-01` - add pytorch hub.
* `2020-07-31` - support ResNet, ResNeXt, CSPResNet, CSPResNeXt. [`r50-pacsp`]() [`x50-pacsp`]() [`cspr50-pacsp`]() [`cspx50-pacsp`]()
* `2020-07-28` - support SAM. [`yolov4-pacsp-sam`]()
* `2020-07-24` - update api.
* `2020-07-23` - support CUDA accelerated Mish activation function.
* `2020-07-19` - support and training tiny YOLOv4. [`yolov4-tiny`]()
* `2020-07-15` - design and training conditional YOLOv4. [`yolov4-pacsp-conditional`]()
* `2020-07-13` - support MixUp data augmentation.
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
* `2020-05-08` - design and training YOLOv4 with FPN neck. [`yolov4-yospp`]()
* `2020-05-01` - training YOLOv4 with Leaky activation function using PyTorch. [`yolov4-paspp`]()

</details>

## Pretrained Models & Comparison

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | yaml | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4**<sub>s-mish</sub> | 672 | 40.3% | 59.4% | 43.8% | 23.9% | 45.3% | 52.2% | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/yolov4s-mish.yaml) | [weights](https://drive.google.com/file/d/1Ku41qm7sLk3vRWI46MslbAMu9pxlqtnk/view?usp=sharing) |
| **YOLOv4**<sub>m-mish</sub> | 672 | 44.7% | 64.0% | 48.7% | 28.3% | 50.2% | 57.7% | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/yolov4m-mish.yaml) | [weights](https://drive.google.com/file/d/1EqbLcdLxjigvYdLhl-YQtPl2qR2KP9iU/view?usp=sharing) |
| **YOLOv4**<sub>l-mish</sub> | 672 | 48.1% | 66.8% | 52.6% | 31.9% | 53.3% | 61.0% | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/yolov4l-mish.yaml) | [weights](https://drive.google.com/file/d/1qzH5DhxUhjoQos3zRd8YFGItEAxWi32X/view?usp=sharing) |
| **YOLOv4**<sub>x-mish</sub> | 672 | **49.8%** | **68.4%** | **54.4%** | **32.7%** | **55.3%** | **63.6%** | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/yolov4x-mish.yaml) | [weights](https://drive.google.com/file/d/1v3xhTxze44VHq_kO7WhATVIkUq0bSGvF/view?usp=sharing) |
|  |  |  |  |  |  |  |
| **YOLOv4**<sub>x-mish</sub> | TTA | **51.2%** | **69.1%** | **56.1%** | **35.6%** | **56.3%** | **64.9%** | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/yolov4x-mish.yaml) | [weights](https://drive.google.com/file/d/1v3xhTxze44VHq_kO7WhATVIkUq0bSGvF/view?usp=sharing) |
|  |  |  |  |  |  |  |
| **CSP**<sub>p6-mish</sub> | 1280 | **53.9%** | **72.0%** | **59.0%** | **39.3%** | **58.3%** | **66.6%** | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/csp-p6-mish.yaml) | - |
| **CSP**<sub>p6-mish</sub> | TTA | **54.4%** | **72.3%** | **59.6%** | **39.8%** | **58.9%** | **67.6%** | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/csp-p6-mish.yaml) | - |
|  |  |  |  |  |  |  |
| **CSP**<sub>p7-mish</sub> | 1536 | **55.0%** | **72.9%** | **60.2%** | **39.8%** | **59.9%** | **68.4%** | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/csp-p7-mish.yaml) | - |
| **CSP**<sub>p7-mish</sub> | TTA | **55.5%** | **72.9%** | **60.8%** | **41.1%** | **60.3%** | **68.9%** | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5/models/csp-p7-mish.yaml) | - |
|  |  |  |  |  |  |  |

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **CSP**<sub>p6-mish</sub> | 1280 | **54.3%** | **72.3%** | **59.5%** | **36.6%** | **58.2%** | **65.5%** | 30 fps |
| **CSP**<sub>p6-mish</sub> | TTA | **54.9%** | **72.6%** | **60.2%** | **37.4%** | **58.8%** | **66.7%** | - |
|  |  |  |  |  |  |  |
| **CSP**<sub>p7-mish</sub> | 1536 | **55.4%** | **73.3%** | **60.7%** | **38.1%** | **59.5%** | **67.4%** | 15 fps |
| **CSP**<sub>p7-mish</sub> | TTA | **55.8%** | **73.2%** | **61.2%** | **38.8%** | **60.1%** | **68.2%** | - |
|  |  |  |  |  |  |  |

## Requirements

```
pip install -r requirements.txt
```

## Training

```
python train.py --data coco.yaml --cfg yolov4l-mish.yaml --weights ''
```
※ Please also install https://github.com/thomasbrandon/mish-cuda

## Testing

```
python test.py --img 672 --conf 0.001 --batch 32 --data coco.yaml --weights weights/yolov4l-mish.pt
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
