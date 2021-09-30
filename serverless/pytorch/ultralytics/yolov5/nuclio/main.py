# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
import numpy as np
from model_handler import ModelHandler
import debugpy
debugpy.listen(5678)

def init_context(context):
    context.logger.info("Init context...  0%")

    model = ModelHandler()
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data['image']))
    image = Image.open(buf).convert('RGB')

    polygon = context.user_data.model.handle(image)
    return context.Response(body=json.dumps(polygon),
                            headers={},
                            content_type='application/json',
                            status_code=200)
