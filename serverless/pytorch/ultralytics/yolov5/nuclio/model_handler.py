# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import torch
import torch.nn as nn
# from dataloaders import helpers
from models.common import Conv
from models.yolo import Model
from utils.postprocess import scale_boxes, non_max_suppression

# def convert_mask_to_polygon(mask):
#     mask = np.array(mask, dtype=np.uint8)
#     cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
#     contours = None
#     if int(cv2.__version__.split('.')[0]) > 3:
#         contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
#     else:
#         contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]
#
#     contours = max(contours, key=lambda arr: arr.size)
#     if contours.shape.count(1):
#         contours = np.squeeze(contours)
#     if contours.size < 3 * 2:
#         raise Exception('Less then three point have been detected. Can not build a polygon.')
#
#     polygon = []
#     for point in contours:
#         polygon.append([int(point[0]), int(point[1])])
#
#     return polygon

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
    from serverless.pytorch.ultralytics.yolov5.nuclio.models.yolo import Detect, Model

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

        # Number of input channels (RGB + heatmap of IOG points)
        # self.net = Network(nInputChannels=5, num_classes=1, backbone='resnet101',
        #     output_stride=16, sync_bn=None, freeze_bn=False)
        #
        # pretrain_dict = torch.load(model_path)
        # self.net.load_state_dict(pretrain_dict)
        # self.net.to(self.device)
        # self.net.eval()
        # w = weights[0] if isinstance(weights, list) else weights
        self.net = Model(cfg='./yolov5m.yaml')

        pretrained_dict = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def handle(self, image):
        with torch.no_grad():
            # extract a crop with padding from the image
            # pred = np.transpose(output_mask.data.numpy()[0, :, :, :], (1, 2, 0))
            # pred = pred > threshold
            # pred = np.squeeze(pred)
            cv_image = np.array(image)
            cv_image = open_cv_image[:, :, ::-1]
            cv_image = np.transpose(cv_image, (2, 0, 1)).astype(np.float32)
            cv_image = np.expand_dims(cv_image, axis=0)
            cv_image /= 255.0
            pred = self.net(torch.Tensor(cv_image).to(self.device))[0]

            pred = non_max_suppression(pred)
            pred = scale_boxes(pred, image)
            return 1

