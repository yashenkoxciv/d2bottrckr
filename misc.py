import cv2
import numpy as np


def draw_detections(frame, detections):
    for detection in detections:
        c1, c2 = detection.y1x1y2x2[:2], detection.y1x1y2x2[2:]
        cv2.rectangle(frame, c1, c2, (235, 164, 52), 2)


def draw_targets(frame, tracker):
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1 #1.5
    TEXT_THICKNESS = 1
    for target_idx in tracker.targets:
        target = tracker.targets[target_idx]
        detection = target.last_detection
        wf = detection.width // 6
        radius = wf if wf > 25 else 25
        if target.unexpected:
            cv2.circle(frame, detection.center, radius, (118, 52, 217), -1)
        else:
            cv2.circle(frame, detection.center, radius, (52, 235, 94), -1)
        
        text = str(target.idx)
        text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(detection.center[0] - text_size[0] / 2), int(detection.center[1] + text_size[1] / 2))

        cv2.putText(
            frame,
            text,
            text_origin,
            TEXT_FACE, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)


""" def draw_detections(frame, detections):
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1 #1.5
    TEXT_THICKNESS = 2
    for detection in detections:
        wf = detection.width // 5
        radius = wf if wf > 20 else 20
        cv2.circle(frame, detection.center, radius, (52, 235, 94), -1)

        text = str(detection.cat)
        text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(detection.center[0] - text_size[0] / 2), int(detection.center[1] + text_size[1] / 2))

        cv2.putText(
            frame,
            text,
            text_origin,
            TEXT_FACE, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA) """


""" def draw_targets(frame, targets, tracks):
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1 #1.5
    TEXT_THICKNESS = 2
    for target in targets:
        center = tracks[target.idx][-1]
        wf = target.width // 6 # 5
        radius = wf if wf > 25 else 25
        cv2.circle(frame, center, radius, (52, 235, 94), -1)

        text = str(target.idx)
        text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))

        cv2.putText(
            frame,
            text,
            text_origin,
            TEXT_FACE, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA) """


def draw_tracker(frame, tracker):
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1 #1.5
    TEXT_THICKNESS = 2
    for target_idx in tracker.frames[-1]:
        detection = tracker.tracks[target_idx][-1]
        center = detection.center
        wf = detection.width // 6 # 5
        radius = wf if wf > 25 else 25
        cv2.circle(frame, center, radius, (52, 235, 94), -1)

        text = str(target_idx)
        text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))

        cv2.putText(
            frame,
            text,
            text_origin,
            TEXT_FACE, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)


def draw_bbx_with_cats(frame, cats, bbxs):
    for c, bbx in zip(cats, bbxs):
        cv2.rectangle(
            frame,
            (bbx[0], bbx[1]), (bbx[2], bbx[3]),
            (255, 56, 56),
            2
        )
        cv2.putText(
            frame,
            str(c),
            (bbx[0], bbx[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1, cv2.LINE_AA
        )


def draw_centers_with_cats(frame, cats, cbbxs):
    for c, bbx in zip(cats, cbbxs):
        cv2.circle(
            frame,
            bbx,
            8,
            (52, 235, 94), -1)
        cv2.putText(
            frame,
            str(c),
            bbx,
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1, cv2.LINE_AA
        )


def centers_of_bbxs(bbxs):
    c = []
    for bbx in bbxs:
        x = (bbx[1] + bbx[3]) // 2
        y = (bbx[0] + bbx[2]) // 2
        c.append((y, x))
    return c # np.array(
