# YOLOv4

This is PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet) which is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

* [[original Darknet implementation of YOLOv4]](https://github.com/AlexeyAB/darknet)

* [[ultralytics/yolov3 based PyTorch implementation of YOLOv4]](https://github.com/WongKinYiu/PyTorch_YOLOv4).

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

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | yaml | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4**<sub>pacsp-s-mish</sub> | 672 | 40.4% | 59.0% | 43.8% | 23.5% | 45.0% | 53.6% | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5_mish_preview/models/yolov4-pacsp-s-mish.yaml) | [weights](https://drive.google.com/file/d/1G4T79LAul7mOlzCBz0XqSAEU0cTsSBRs/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-mish</sub> | 672 | 47.2% | 65.6% | 51.3% | 29.6% | 52.5% | 60.8% | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5_mish_preview/models/yolov4-pacsp-mish.yaml) | [weights](https://drive.google.com/file/d/1vMjsWEoYh-8Vnicn9YCskjjBK0hIKaBE/view?usp=sharing) |
| **YOLOv4**<sub>pacsp-mish</sub> | 672 | 48.6% | 67.1% | 52.8% | 32.1% | 54.1% | 63.1% | [yaml](https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/u5_mish_preview/models/yolov4-pacsp-x-mish.yaml) | [weights]() |
|  |  |  |  |  |  |  |

## Requirements

```
pip install -r requirements.txt
```

## Training

```
python train.py --data coco.yaml --cfg yolov4-pacsp-mish.yaml --weights ''
```
※ Please also install https://github.com/thomasbrandon/mish-cuda

## Testing

```
python test.py --img 672 --conf 0.001 --batch 32 --data coco.yaml --weights weights/yolov4-pacsp-mish.pt
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
