
## Image Processing Detector

### Objective

The objective of the image processing detector is to use NIR images collected from the JAI camera to apply thresholding and masking and help label RGB images.

The workflow would be:
1. Upload RGB image to CVAT
2. Apply image processing detector using the automatic annotaion (magic wand) tool
3. Annotator applys thresholding on the corresponding NIR image of the RGB image.
4. A mask is returned by the tool and applied on the RGB image.

### Achievements

The written code is at the moment able to achieve the following:

1. Deploy the image processing detector as nuclio docker container.
2. Read an arbitrary NIR image inside the nuclio container
3. Apply thresholding on this NIR image
4. Prepare a polygon
5. Send the polygon to CVAT

#### Explanation of Code

##### `detector.py`

The detector takes as an argument an image and return an image, as a numpy array, with a mask applied.

##### `masktopolygon.py`

Given a numpy array of a mask applied to an RGB image, the only function in this file returns a polygon which is appropriate for drawing in CVAT.

#### `modelhandly.py`, `main.py` and `function.yml`

These files are in accordance with the template of other models provided. The import the detector function and build a docker container with the provided files.

### To Do

#### Mount Storage of NIR Image

For the Nuclio Image Processing Docker Container to apply masking on the NIR image, it has to have access to them. For this, we need to mount a storage of NIR image, corresponding to the RGB images, inside the container.

#### Pass Filename to the Image Processing Detector

The Nuclio Image Processing container recieves an RGB Image and needs to apply thresholding to the NIR image. For this it needs a matching mechanism for finding the right NIR image for the given RGB image. The filename of the image is not passed by default to Nuclio and needs to be added. Right now the system works by using an arbitrary image the path of which is hardcoded in the thresholding function.

#### Change Mask to Polygon Conversion Technique

The conversion technique availble in the `masktopolygon.py` file could be different. This can be achieved by changing the implementation of the `convert_mask_to_polygon` function in the file.

