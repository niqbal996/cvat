
from detector import thresholdimage

from masktopolygon import convert_mask_to_polygon

def masktopolygon(mask):
    mask = np.array(mask, dtype=np.uint8)
    cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less than three points have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon

def fetchnirimage(filename):
    pass

class ModelHandler:
    def __init__ (self):
        self.applythreshold = thresholdimage

    def infer (self, image):
        mask = self.applythreshold(image)
        polygon = convert_mask_to_polygon(mask)

        #return ([[50, 50, 150, 150, 250, 250, 50, 250]], [1], [0])
        return (polygon, [0])

