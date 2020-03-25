import cv2
import numpy as np
from mask import Mask
from enum import Enum
from detector import Detection
from collections import defaultdict, namedtuple
from sklearn.metrics.pairwise import euclidean_distances


class Status(Enum):
    APPEARED, APPEARED_UNEXPECTEDLY, STANDING, WALKING, HIDING, REAPPEARED, LEAVING, VANISHED = range(8)


class Target:
    def __init__(self, idx, detection, unexpected=False):
        self.idx = idx
        self.track = [detection]
        self.states = [] # or add field to Detection
        self.unexpected = unexpected
    
    def extend_track(self, detection):
        self.track.append(detection)
    
    @property
    def last_detection(self):
        return self.track[-1]
    
    def __str__(self):
        return '<Target {0:04d} {1}->{2} len={3} unexpected={4}>'.format(
            self.idx,
            self.track[0].center,
            self.track[-1].center,
            len(self.track),
            self.unexpected
        )
    
    def __repr__(self):
        return str(self)


class Identifier:
    def __init__(self, start=0):
        self.counter = start
        self.start = start
    
    @property
    def new_one(self):
        value = self.counter
        self.counter += 1
        return value
    
    @property
    def previous(self):
        return self.counter if self.counter == self.start else self.counter - 1
    
    def __str__(self):
        return '<Identifier {0:06d}>'.format(self.previous)
    
    def __repr__(self):
        return str(self)


MatchItem = namedtuple('MatchItem', ['match', 'min_dist', 'detection_idx']) # , verbose=True


class Tracker:
    def __init__(self, distance_threshold, depthmask_file, scenemask_file, scenemask_threshold, frame_threshold):
        self.idxm = Identifier()
        self.targets = {}
        self.lost_targets = {}
        #self.last_targets = set()
        # thresholds
        self.distance_threshold = distance_threshold
        self.depthmask = cv2.imread(depthmask_file).mean(2) / 255.0
        self.scenemask = Mask(scenemask_file, scenemask_threshold)
        self.frame_threshold = frame_threshold
    
    """ def initialize(self, detections):
        self.frames.append([])
        for detection in detections:
            self.register(detection) """
    
    @staticmethod
    def euclidean_distance(c1, c2):
        dist = np.sqrt(np.power(c1[0] - c2[0], 2) + np.power(c1[1] - c2[1], 2))
        return dist
    
    
    def create_target(self, detection):
        target = Target(
            self.idxm.new_one,
            detection,
            unexpected=not self.scenemask.check(*detection.center)
        )
        self.targets[target.idx] = target
        return target
    

    def candidates(self):
        pass


    def filter_detections(self, frames_detections):
        for frame_idx in range(len(frames_detections)):
            detections = frames_detections[frame_idx]
            detection_idxs = range(len(detections))
            to_delete_idxs = set()
            for detection1_idx in detection_idxs:
                d1 = detections[detection1_idx]
                for detection2_idx in detection_idxs:
                    d2 = detections[detection2_idx]
                    depth_k = self.depthmask[d1.center[1], d1.center[0]] #/ 1.3 #1 #self.depthmask[d1.center[1], d1.center[0]] / 1.3
                    if (Tracker.euclidean_distance(d1.center, d2.center) < self.distance_threshold*depth_k) and (detection1_idx != detection2_idx):
                        to_delete_idxs.add(detection1_idx)
                        to_delete_idxs.add(detection2_idx)
            filtered_detections = []
            for detection_idx in detection_idxs:
                if detection_idx not in to_delete_idxs:
                    filtered_detections.append(detections[detection_idx])
            frames_detections[frame_idx] = filtered_detections

    
    def update(self, detections):
        if not self.targets:
            for detection in detections:
                target = self.create_target(detection)
        else:
            candidates = {}
            for target_idx in self.targets:
                target = self.targets[target_idx]
                candidates[target_idx] = {'min_dist': np.inf, 'detection_idx': None}
                detection_idxs = set(range(len(detections)))
                for detection_idx in detection_idxs:
                    dist = Tracker.euclidean_distance(
                        target.last_detection.center,
                        detections[detection_idx].center
                    )
                    #import ipdb; ipdb.set_trace()
                    if (dist < self.distance_threshold) and (dist < candidates[target_idx]['min_dist']):
                        candidates[target_idx]['min_dist'] = dist
                        candidates[target_idx]['detection_idx'] = detection_idx
            # process candidates 
            match = defaultdict(list)
            for target_idx in candidates:
                detected_candidate_idx = candidates[target_idx]['detection_idx']
                if detected_candidate_idx is None: # lost target (delete from targets; move to lost targets)
                    self.lost_targets[target_idx] = self.targets[target_idx]
                    del self.targets[target_idx]
                else:
                    match[detected_candidate_idx].append(target_idx)
            # form unused and crowds set
            crowds = []
            for detected_candidate_idx in match:
                if len(match[detected_candidate_idx]) > 1: # crowd detected
                    crowds.append({
                        'targets': match[detected_candidate_idx],
                        'detection': detected_candidate_idx
                    })
                    #import ipdb; ipdb.set_trace()
                    #print(crowds)
                else: # extend track
                    target_idx = match[detected_candidate_idx][0]
                    self.targets[target_idx].extend_track(detections[detected_candidate_idx])
            used_detections = match.keys()
            unused_detections = detection_idxs - used_detections
            # process !unused detections! and !crowds!
            # TODO: what to do with crowds????
            # create new targets for unused detections
            for unused_detection_idx in unused_detections:
                detection = detections[unused_detection_idx]
                min_f_dist = np.inf
                min_lost_target_idx = None
                for lost_target_idx in self.lost_targets: # find match for (unused) detection among lost_targets
                    lost_target = self.lost_targets[lost_target_idx]
                    last_detection = lost_target.last_detection
                    xy_dist = Tracker.euclidean_distance(
                        lost_target.last_detection.center,
                        detection.center
                    )
                    if ((detection.frame - self.frame_threshold) <= last_detection.frame) and (xy_dist <  self.distance_threshold):
                        # TODO: use reid threshold ???
                        f_dist = np.mean(np.power(detection.features - last_detection.features, 2))
                        if f_dist < min_f_dist:
                            min_f_dist = f_dist
                            min_lost_target_idx = lost_target_idx
                if min_lost_target_idx is not None: # extend lost track; and revive it
                    lost_target = self.lost_targets[min_lost_target_idx]
                    lost_target.extend_track(detection)
                    lost_target.unexpected = False # !!!!!!!!!! DANGER !!!!!!!!!!
                    self.targets[lost_target.idx] = lost_target
                    del self.lost_targets[min_lost_target_idx]
                    #print('R:', target, end=' ')
                else: # create new target or match with lost targets
                    target = self.create_target(detection)
                    """ unexpected = not self.scenemask.check(*detection.center)
                    if unexpected:
                        min_f_dist = np.inf
                        min_lost_target_idx = None
                        for lost_target_idx in self.lost_targets: # find match for (unused) detection among lost_targets
                            lost_target = self.lost_targets[lost_target_idx]
                            last_detection = lost_target.last_detection
                            f_dist = np.mean(np.power(detection.features - last_detection.features, 2))*1000
                            if ((detection.frame - 90) > last_detection.frame) and (f_dist < 0.75):
                                # TODO: use reid threshold ???
                                if f_dist < min_f_dist:
                                    min_f_dist = f_dist
                                    min_lost_target_idx = lost_target_idx
                        if min_lost_target_idx is not None: # extend track
                            lost_target = self.lost_targets[min_lost_target_idx]
                            lost_target.extend_track(detection)
                            lost_target.unexpected = False # !!!!!!!!!! DANGER !!!!!!!!!!
                            self.targets[lost_target.idx] = lost_target
                            del self.lost_targets[min_lost_target_idx]
                            #print('R:', target, end=' ')
                        else:
                            target = self.create_target(detection)
                    else:
                        target = self.create_target(detection) """
                    #if target.idx == 46:
                    #    import ipdb; ipdb.set_trace()
                    #print('C:', target)
                #print()
            #import ipdb; ipdb.set_trace()

        #import ipdb; ipdb.set_trace()

