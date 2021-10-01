# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
# from dataloaders import helpers
from models.common import Conv
from utils.postprocess import non_max_suppression, scale_coords

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output



def attempt_load(weights, map_location=None, inplace=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble

class ModelHandler:
    def __init__(self):
        base_dir = os.environ.get("MODEL_PATH", "/opt/nuclio/yolov5")
        model_path = os.path.join(base_dir, "yolov5.pt")
        self.device = torch.device("cpu")
        self.net = attempt_load(model_path, map_location=self.device)  # load FP32 model
        self.net.eval()
        self.names = self.net.module.names if hasattr(self.net, 'module') else self.net.names # get class names
        self.names[0] = 'Maize'
        self.names[1] = 'Weed'

    def handle(self, image):
        with torch.no_grad():
            cv_image = np.asarray(image)
            orig_size = cv_image.shape[0:2]
            cv_image = cv2.resize(cv_image, (380, 640), interpolation=cv2.INTER_NEAREST)
            # cv_image = cv2.cvtColor(np.array(cv_image), cv2.COLOR_BGR2RGB)
            cv_image = np.transpose(cv_image, (2, 0, 1)).astype(np.float32) # channels first
            cv_image /= 255.0
            cv_image = np.expand_dims(cv_image, axis=0)
            pred = self.net(torch.from_numpy(cv_image).to(self.device))[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
            polygons = []
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(cv_image.shape[2:], det[:, :4], orig_size).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        polygons.append({"confidence": str(float(conf.cpu())),
                                        "label": self.names[int(cls)],
                                        "points": [int(np.array(i.cpu())) for i in xyxy],
                                        "type": "rectangle",
                                        })
            # polygons.append({ "confidence": str(1),"label": str("maize"),"points": [10, 200, 100, 456],"type": "rectangle"})
            return polygons

