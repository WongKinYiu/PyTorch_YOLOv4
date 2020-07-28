# YOLOv4

This is PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet) which is based on [ultralytics/yolov3](https://github.com/ultralytics/yolov3).

* [[original Darknet implementation of YOLOv4]](https://github.com/AlexeyAB/darknet)

* [[ultralytics/yolov5 based PyTorch implementation of YOLOv4]](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5_preview).

### development log

<details><summary> <b>Expand</b> </summary>
  
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

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4**<sub>paspp</sub> | 736 | 45.7% | 64.2% | 50.3% | 27.4% | 51.3% | 58.6% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-paspp.cfg) | [weights](https://drive.google.com/file/d/1FraA4vmlBh5RoQB7ZGVc01UyCgxSlbpO/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-s</sub> | 736 | 36.0% | 54.2% | 39.4% | 18.7% | 41.2% | 48.0% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-s.cfg) | [weights](https://drive.google.com/file/d/1saE6CEvNDPA_Xv34RdxYT4BbCtozuTta/view?usp=sharing) |
| **YOLOv4**<sub>pacsp</sub> | 736 | 46.4% | 64.8% | 51.0% | 28.5% | 51.9% | 59.5% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp.cfg) | [weights](https://drive.google.com/file/d/1SPCjPnMgA8jlfIGsAnFsMPdJU8dJeo7E/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-x</sub> | 736 | **47.6%** | **66.1%** | **52.2%** | **29.9%** | **53.3%** | **61.5%** | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-x.cfg) | [weights](https://drive.google.com/file/d/1MtwO5tvXvvyloc12-wZ2lMBzGKd9hsof/view?usp=sharing) |
|  |  |  |  |  |  |  |
| **YOLOv4**<sub>pacsp-s-mish</sub> | 736 | 37.4% | 56.3% | 40.0% | 20.9% | 43.0% | 49.3% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-s-mish.cfg) | [weights](https://drive.google.com/file/d/1Gmy2Q6af1DQ5CAb6415cVFkIgtOIt9xs/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-mish</sub> | 736 | 46.5% | 65.7% | 50.2% | 30.0% | 52.0% | 59.4% | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-mish.cfg) | [weights](https://drive.google.com/file/d/10pw28weUtOceEexRQQrdpOjxBb79sk3u/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-x-mish</sub> | 736 | **48.5%** | **67.4%** | **52.7%** | **30.9%** | **54.0%** | **62.0%** | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-pacsp-x-mish.cfg) | [weights](https://drive.google.com/file/d/1GsLaQLfl54Qt2C07mya00S0_FTpcXBdy/view?usp=sharing) |
|  |  |  |  |  |  |  |
| **YOLOv4**<sub>tiny</sub> | 416 | **22.5%** | **39.3%** | **22.5%** | **7.4%** | **26.3%** | **34.8%** | [cfg](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/cfg/yolov4-tiny.cfg) | [weights](https://drive.google.com/file/d/1aQKcCvTAl1uOWzzHVE9Z8Ixgikc3AuYQ/view?usp=sharing) |
|  |  |  |  |  |  |  |

## Requirements

```
pip install -r requirements.txt
```
※ For running Mish models, please install https://github.com/thomasbrandon/mish-cuda

## Training

```
python train.py --data coco2017.data --cfg yolov4-pacsp.cfg --weights '' --name yolov4-pacsp --img 640 640 640
```

## Testing

```
python test_half.py --data coco2017.data --cfg yolov4-pacsp.cfg --weights yolov4-pacsp.pt --img 736 --iou-thr 0.7 --batch-size 8
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
