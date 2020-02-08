import cv2

def cv2_imread_rgb(uri):
    return cv2.cvtColor(cv2.imread(str(uri)), cv2.COLOR_BGR2RGB)

