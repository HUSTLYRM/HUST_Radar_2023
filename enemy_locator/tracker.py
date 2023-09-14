from random import randint

import cv2


class Tracker_IoU:
    def __init__(self, thresh,):
        self.thresh = thresh
        pass

    def track(self, targets_last_frame, targets_curr_frame):
        pass


class TrackerPool:
    def __init__(self, lost_thresh=5):
        self.trackers = [cv2.legacy.TrackerMIL.create() for _ in range(5)]
        self.lost_count = [0 for _ in range(5)]
        self.lost_thresh = lost_thresh

        self.is_tracked = [False for _ in range(5)]
        self.tracker_conf = [0 for _ in range(5)]
        self.colors = [(randint(64, 255), randint(64, 255), randint(64, 255)) for _ in range(5)]

    def init_tracker_for(self, idx, frame, rect):
        self.trackers[idx].init(frame, rect)
        self.is_tracked[idx] = True

    def get_pred_rect_for(self, idx, frame):
        _, prerd_rect = self.trackers[idx].update(frame)
        if prerd_rect is None:
            self.lost_count[idx] += 1
            if self.lost_count[idx] > self.lost_thresh:
                self.remove_tracker_for(idx)
            return None
        else:
            self.lost_count[idx] = 0
            return prerd_rect

    def remove_tracker_for(self, idx):
        self.trackers[idx] = None
        self.lost_count[idx] = 0
        self.is_tracked[idx] = False

    def set_tracker_for(self, idx, frame, rect):
        self.trackers[idx] = cv2.legacy.TrackerMIL.create()
        self.init_tracker_for(idx, frame, rect)

    def display_all(self, frame, boxes):
        for i, box in enumerate(boxes):
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, self.colors[i], 2, 1)
        cv2.imshow('Trackers', frame)

    def display_part(self, frame, boxes, ids):
        for i, box in enumerate(boxes):
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, self.colors[ids[i]], 2, 1)
        cv2.imshow('Trackers', frame)
