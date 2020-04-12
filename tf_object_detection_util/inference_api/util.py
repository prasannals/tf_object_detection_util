import cv2

def cv2_imread_rgb(uri:'str or pathlib.Path') -> 'np.ndarray':
    '''
    reads an image at the URI specified in "uri" as an RGB image 
    returns the output as a numpy ndarray of shape (height, width, 3)
    '''
    return cv2.cvtColor(cv2.imread(str(uri)), cv2.COLOR_BGR2RGB)

