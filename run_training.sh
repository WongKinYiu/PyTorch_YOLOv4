# !/bin/bash
python train.py --device 0 --batch-size 16 --img 640 640 --data data/coco128.yaml --cfg cfg/yolov4-pacsp.cfg --weights 'weights/yolov4_pacsp.pt' --name yolov4-pacsp --epochs 100

