import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # Maximum number of frames to keep a track without updates
        self.min_hits = min_hits  # Minimum number of hits (detections) to start a track
        self.iou_threshold = iou_threshold  # IOU threshold for matching detections and tracks

        self.tracks = []
        self.track_count = 0

    def update(self, detections):
        # Run the Kalman filter and update track state for all existing tracks
        for track in self.tracks:
            track.predict()
            track.update()

        # Create new tracks for unassigned detections
        unassigned_tracks = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update < 1:
                continue
            unassigned_tracks.append(track)
            track.mark_missed()

        # Associate unassigned tracks with detections using Hungarian algorithm
        unassigned_detections = np.ones(len(detections), dtype=bool)
        iou_matrix = np.zeros((len(unassigned_tracks), len(detections)))
        for i, track in enumerate(unassigned_tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = track.iou(detection)
        matched_indices = linear_sum_assignment(-iou_matrix)

        # Update existing tracks with assigned detections
        for i, j in zip(*matched_indices):
            track = unassigned_tracks[i]
            detection = detections[j]
            track.update_assignment(detection)
            unassigned_detections[j] = False

        # Create new tracks for unassigned detections
        for j in range(len(detections)):
            if not unassigned_detections[j]:
                continue
            detection = detections[j]
            track = Track(detection)
            self.tracks.append(track)
            self.track_count += 1

        # Remove lost tracks
        self.tracks = [track for track in self.tracks if track.is_confirmed()]

        # Increment age and mark missed for non-updated tracks
        for track in self.tracks:
            track.increment_age()
            if not track.is_updated():
                track.mark_missed()

        # Return updated tracks
        return [track.to_tlwh() for track in self.tracks if track.is_confirmed()]


class Track(object):
    def __init__(self, detection):
        self.track_id = None
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.box = detection

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])
        self.kf.R = np.diag([10.0, 10.0, 10.0, 10.0])
        self.kf.Q = np.eye(8) * 0.01
        self.kf.x[:4] = self.to_xyah()

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self):
        self.hits += 1
        self.time_since_update = 0
        self.kf.update(self.to_xyah())

    def update_assignment(self, detection):
        self.box = detection
        self.update()

    def increment_age(self):
        self.age += 1

    def mark_missed(self):
        self.time_since_update += 1

    def is_confirmed(self):
        return self.hits >= self.min_hits

    def is_updated(self):
        return self.time_since_update == 0

    def to_tlwh(self):
        x, y, a, h = self.kf.x[:4]
        w = h * a
        return np.array([x - w / 2, y - h / 2, w, h])

    def to_xyah(self):
        x, y, w, h = self.box
        a = w / h
        return np.array([x + w / 2, y + h / 2, a, h])

    def iou(self, detection):
        box = self.to_tlwh()
        detection = Track(detection).to_tlwh()
        x1 = max(box[0], detection[0])
        y1 = max(box[1], detection[1])
        x2 = min(box[0] + box[2], detection[0] + detection[2])
        y2 = min(box[1] + box[3], detection[1] + detection[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box[2] * box[3]
        area2 = detection[2] * detection[3]
        union = area1 + area2 - intersection
        iou = intersection / union
        return iou
