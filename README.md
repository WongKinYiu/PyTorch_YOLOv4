# YOLOv4

This is PyTorch implementation of [YOLOv4](https://github.com/AlexeyAB/darknet) which is based on [ultralytics/yolov3](https://github.com/ultralytics/yolov3).

### development log

* `2020-06-12` - follow [ultralytics](https://github.com/ultralytics/yolov5) to design scaled YOLOv4. [`yolov4-pacsp-s`]() [`yolov4-pacsp-m`]() [`yolov4-pacsp-l`]() [`yolov4-pacsp-x`]()
* `2020-06-07` - design scaling methods for CSP-based models. [`yolov4-pacsp-25`]() [`yolov4-pacsp-75`]()
* `2020-06-03` - update COCO2014 to COCO2017.
* `2020-05-30` - update FPN neck to CSPFPN. [`yolov4-yocsp`]() [`yolov4-yocsp-mish`]()
* `2020-05-24` - update neck of YOLOv4 to CSPPAN. [`yolov4-pacsp`]() [`yolov4-pacsp-mish`]()
* `2020-05-15` - training YOLOv4 with Mish activation function. [`yolov4-yospp-mish`]() [`yolov4-paspp-mish`]()
* `2020-05-08` - design and training YOLOv4 with FPN neck. [`yolov4-yospp`]()
* `2020-05-01` - training YOLOv4 with Leaky activation function using PyTorch. [`yolov4-paspp`]()

## Pretrained Models & Comparison

| Model | Size | AP<sup>val</sup> | AP<sup>test</sup> | Batch 32 FPS | cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4**<sub>pacsp-mish</sub> | 608 | (45.9%) | (45.7%) | 104 | [cfg]() | [weights]() |
| **YOLOv4**<sub>paspp-mish</sub> | 608 | (45.0%) | (44.9%) |  | [cfg]() | [weights]() |
| **YOLOv4**<sub>yocsp-mish</sub> | 608 | (44.7%) | (44.5%) |  | [cfg]() | [weights]() |
| **YOLOv4**<sub>yospp-mish</sub> | 608 | (44.3%) | (44.1%) |  | [cfg]() | [weights]() |
| **YOLOv4**<sub>pacsp</sub> | 608 | 45.4% | 45.2% | 125 | [cfg]() | [weights]() |
| **YOLOv4**<sub>paspp</sub> | 608 | 44.4% | 44.6% |  | [cfg]() | [weights]() |
| **YOLOv4**<sub>yocsp</sub> | 608 | 44.3% | 44.2% |  | [cfg]() | [weights]() |
| **YOLOv4**<sub>yospp</sub> | 608 | 43.9% | 43.6% |  | [cfg]() | [weights]() |
|  |  |  |  |  |  |  |
| **YOLOv5**<sub>s</sub> | 736 | 35.5 | 35.5 | 223 | - | - |
| **YOLOv5**<sub>m</sub> | 736 | 42.7 | 42.7 | 150 | - | - |
| **YOLOv5**<sub>l</sub> | 736 | 45.7 | 45.9 | 98 | - | - |
| **YOLOv5**<sub>x</sub> | 736 | 47.2 | 47.3 | 59 | - | - |
* (%) means trained on COCO2014.
* FPS is tested on Tesla V100 including pre-processing, FP32 model inference, and post-processing. 

## Requirements

```
pip install -r requirements.txt
```

## Training

```
python train.py --data coco2017.data --cfg yolov4.cfg --weights '' --name yolov4 --img 448 768 512 --batch 16 --device 0
```

## Testing

```
python test.py --data coco2017.data --cfg yolov4.cfg --weights yolov4.pt --img 608 --iou-thr 0.7 --batch-size 32 --device 0
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
