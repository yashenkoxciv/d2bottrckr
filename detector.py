import numpy as np
from reid import ReIdentification
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detection:
    def __init__(self, bbx, cat, features, frame):
        self.x1, self.x2 = bbx[1], bbx[3]
        self.y1, self.y2 = bbx[0], bbx[2]
        self.y1x1y2x2 = tuple(bbx)
        self.cat = cat
        self.features = features
        self.frame = frame
    
    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1
    
    @property
    def center(self):
        x = (self.x1 + self.x2) // 2
        y = (self.y1 + self.y2) // 2
        return y, x
    
    def __str__(self):
        return '<Detection cat={0} width={1} height={2} center={3} frame={4}>'.format(
            self.cat, self.width, self.height, self.center, self.frame
        )
    
    def __repr__(self):
        return str(self)


class Detector:
    def __init__(self, config, d2_weights, reid_weights, roi_threshold=0.5, acceptable_cats=[0]): # 0.5
        self.detector_cfg = get_cfg()
        self.detector_cfg.merge_from_file(config)
        self.detector_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_threshold
        self.detector_cfg.MODEL.WEIGHTS = d2_weights
        self.predictor = DefaultPredictor(self.detector_cfg)
        self.acceptable_cats = acceptable_cats
        self.reid = ReIdentification(reid_weights)
        self.frame_num = 0
    
    def __call__(self, image):
        output = self.predictor(image)
        all_cats = output['instances'].pred_classes.cpu().numpy()
        all_bbxs = output['instances'].pred_boxes.tensor.cpu().numpy().astype(np.int)
        detections = []
        for c, bbx in zip(all_cats, all_bbxs):
            if c in self.acceptable_cats:
                features = self.reid(image[bbx[1]:bbx[3], bbx[0]:bbx[2]][:, :, ::-1]) #.copy()
                detections.append(Detection(bbx, c, features, self.frame_num))
        self.frame_num += 1
        return detections







