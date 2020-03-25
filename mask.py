import numpy as np

try:
    import cv2
    backend = 'cv2'
except ImportError:
    try:
        from PIL import Image
        backend = 'pil'
    except ImportError:
        raise Exception('OpenCV or PIL is required')


class Mask:
    """
    Instancies of this class allow to check if the new bounding box appeared or broken.
    """
    def __init__(self, filename, threshold=0):
        if backend == 'cv2':
            self.mask = cv2.imread(filename).mean(2)
        elif backend == 'pil':
            self.mask = np.array(Image.open(filename)).mean(2)
        self.threshold = threshold
    
    def check(self, y, x):
        """This method checks if the point (x, y) is inside mask (white color).

        Parameters
        ----------
        x
            Vertical (row) index of the bounding box.
        y
            Horizontal (column) index of the bounding box.

        Returns
        -------
        bool
            True if mask item is greater than threshold (0 by default), False otherwise.

        """
        return self.mask[x, y] > self.threshold




