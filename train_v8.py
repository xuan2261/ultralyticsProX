import sys
import argparse
import os

sys.path.append(r'E:\GitHubRepo\PR\ultralyticsPro-') # Path

from ultralytics import YOLO

'''
python train_v8.py --cfg 

'''

def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    model = YOLO(yaml) # 直接加载yaml文件训练
    # model = YOLO(weights)  # 直接加载权重文件进行训练
    # model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练



    # print(model)

    model.info()

    results = model.train(data='coco128.yaml', 
                        epochs=200, 
                        imgsz=640, 
                        workers=2, 
                        batch=2,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\v8\yolov8.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)