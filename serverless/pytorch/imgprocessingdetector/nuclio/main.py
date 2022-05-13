
from model_handler import ModelHandler
import json
import io
import base64
from PIL import Image

labels = {0 : "crop", 1 : "weed"}

def init_context(context):
    context.logger.info("Init context...  0%")

    model = ModelHandler()
    context.user_data.model_handler = model.infer

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    print(event)
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)

    predictions = context.user_data.model_handler(image)
    box, label = predictions
    results = []
    for box, label in zip(box, label):
        results.append({
            "confidence": 1,
            "label": labels[label],
            "points": box,
            "type": "polygon"
        })

    return context.Response(body=json.dumps(results),
                            headers={},
                            content_type='application/json',
                            status_code=200)

