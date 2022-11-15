import json
import base64
import io
from PIL import Image

import torch
import numpy as np
from detectron2.model_zoo import get_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
#from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)

CONFIG_OPTS = ["MODEL.WEIGHTS", "model_final.pth"]
CONFIDENCE_THRESHOLD = 0.5

labels = {1 : "maize", 0 : "weed" }

def init_context(context):
    context.logger.info("Init context...  0%")

    #args = default_argument_parser().parse_args()
    cfg_file = "detectron2/configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py"
    cfg = LazyConfig.load(cfg_file)
    cfg.train.output_dir = "fcos-output/"
    cfg.dataloader.test.num_workers = 2
    default_setup(cfg, None)

    '''
    if torch.cuda.is_available():
        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cuda'])
    else:
        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cpu'])

    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.freeze()
    '''

    model = instantiate(cfg.model)
    # model.to(cfg.train.device)
    # model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    model.eval()

    context.user_data.model_handler = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run fcos-maize model")

    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    # input should be cwh
    image = np.transpose(image, (2, 0, 1))
    # fcos expects a list of input dictionaries, which have the attributes, file_name, height, width, image_id, and image
    # image has to be a pytorch.Tensor object
    # need to use .copy() because image has negative strides which are not supported by pytorch
    image = torch.tensor(image.copy())
    inputs = [{"image" : image}]

    predictions = context.user_data.model_handler(inputs)
    predictions = predictions[0]

    instances = predictions['instances']
    pred_boxes = instances._fields['pred_boxes']
    scores = instances._fields['scores']
    pred_classes = instances._fields['pred_classes']
    results = []
    for box, score, label in zip(pred_boxes, scores, pred_classes):
        label = labels[int(label)]
        if score >= threshold:
            results.append({
                "confidence": str(float(score)),
                "label": label,
                "points": box.tolist(),
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)

