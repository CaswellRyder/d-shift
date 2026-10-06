"""Bounded, observational slow-motion scheduler. Never issues flight commands.

Cached identities expire. Local template association is not identity proof and
must be evaluated on real recordings before use by an autonomous controller.
"""
import time

import cv2
import numpy as np

from .tracking import iou
from .vision import proposals, suppress_duplicates


class TemporalVision:
    def __init__(self, predictor, budget=4, scan_interval=.5, refresh_interval=.6,
                 label_ttl=1.0, candidate_limit=12, background_refresh_interval=1.8,
                 proposal_profile="balloon_components", difference_backend="native"):
        if difference_backend not in ("numpy", "native"):
            raise ValueError("Unsupported difference backend")
        self.difference_backend = difference_backend
        if proposal_profile not in ("balloon_components", "goal_lut"):
            raise ValueError("Unsupported temporal proposal profile")
        self.proposal_profile = proposal_profile
        if (not isinstance(budget,int) or not isinstance(candidate_limit,int)
                or not 1 <= budget <= candidate_limit <= 64):
            raise ValueError("Invalid inference/candidate budget")
        if (not all(np.isfinite(x) for x in (scan_interval,refresh_interval,label_ttl,background_refresh_interval))
                or not 0 < scan_interval <= label_ttl or not 0 < refresh_interval <= label_ttl):
            raise ValueError("Invalid temporal intervals")
        if not refresh_interval <= background_refresh_interval <= 2:
            raise ValueError("Background refresh must be bounded between refresh_interval and 2 seconds")
        self.predictor = predictor
        self.budget, self.limit = budget, candidate_limit
        self.scan_interval, self.refresh_interval, self.ttl = scan_interval, refresh_interval, label_ttl
        self.background_refresh_interval = background_refresh_interval
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.last_scan = -float("inf")
        self.last_time = None
        self.previous_small = None
        self.frame_shape = None

    def difference_exceeds(self, a, b, threshold):
        # Internal signatures are integer-valued float32 images from uint8 resize.
        # Their L1 sum is exact at these sizes. Compare sums, avoiding three NumPy
        # operations and temporary arrays; no relaxed tracking/change threshold.
        if self.difference_backend == "native":
            return cv2.norm(a, b, cv2.NORM_L1) > threshold*a.size
        return np.mean(np.abs(a-b)) > threshold

    @staticmethod
    def signature(rgb, box):
        x1,y1,x2,y2 = map(int, box)
        return cv2.resize(rgb[y1:y2,x1:x2], (12,12), interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def template(gray, box):
        x1,y1,x2,y2 = map(int, box)
        return gray[y1//2:max(y1//2+1,y2//2), x1//2:max(x1//2+1,x2//2)].copy()

    @staticmethod
    def template_box(rgb, box):
        # Include surrounding edges: a uniform colored interior cannot locate motion.
        a,b,c,d = box
        return [max(0,a-6),max(0,b-6),min(rgb.shape[1],c+6),min(rgb.shape[0],d+6)]

    def _scan(self, rgb, gray, now):
        found = proposals(rgb, self.predictor.metadata["task"], self.limit, profile=self.proposal_profile)
        old, self.tracks = self.tracks, []
        used = set()
        for candidate in found:
            choices = [(iou(candidate["box"], row["box"]), index) for index, row in enumerate(old)
                       if index not in used and row["color_group"] == candidate["color_group"]]
            overlap, index = max(choices, default=(0, -1))
            signature = self.signature(rgb, candidate["box"])
            if overlap >= .4:
                used.add(index)
                track = old[index]
                if self.difference_exceeds(signature, track["signature"], 20):
                    track["prediction"] = None
                    track["classified"] = -float("inf")
            else:
                track = dict(track_id=self.next_id, prediction=None, classified=-float("inf"))
                self.next_id += 1
            track.update(candidate)
            context_box = self.template_box(rgb,candidate["crop_box"])
            track.update(signature=signature, template=self.template(gray, context_box),
                         template_box=context_box)
            self.tracks.append(track)
        self.last_scan = now

    def _track(self, rgb, gray):
        kept, lost = [], False
        height, width = rgb.shape[:2]
        for track in self.tracks:
            x1,y1,x2,y2 = track["template_box"]
            important = track["prediction"] is not None and track["prediction"]["accepted"]
            left, top = max(0,x1//2-6), max(0,y1//2-6)
            right, bottom = min(gray.shape[1],x2//2+6), min(gray.shape[0],y2//2+6)
            search, template = gray[top:bottom,left:right], track["template"]
            if min(template.shape) < 2 or any(a < b for a,b in zip(search.shape,template.shape)):
                lost = lost or important
                continue
            scores = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED)
            error, _, position, _ = cv2.minMaxLoc(scores)
            dx, dy = 2*(left+position[0]-x1//2), 2*(top+position[1]-y1//2)
            a,b,c,d = track["box"]
            box = [a+dx,b+dy,c+dx,d+dy]
            if (error > .15 or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height
                    or self.difference_exceeds(self.signature(rgb, box), track["signature"], 20)):
                lost = lost or important
                continue
            track["box"] = box
            track["template_box"] = [x1+dx,y1+dy,x2+dx,y2+dy]
            a,b,c,d = track["crop_box"]
            track["crop_box"] = [max(0,a+dx),max(0,b+dy),min(width,c+dx),min(height,d+dy)]
            kept.append(track)
        self.tracks = kept
        return lost

    def observe(self, rgb, now=None):
        start = time.perf_counter()
        now = time.monotonic() if now is None else float(now)
        if (not np.isfinite(now) or (self.last_time is not None and now <= self.last_time)):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 8:
            raise ValueError("Expected RGB uint8 frame")
        if (self.frame_shape != rgb.shape
                or (self.last_time is not None and now-self.last_time > self.ttl)):
            self.tracks = []
            self.last_scan = -float("inf")
            self.previous_small = None
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                          (rgb.shape[1]//2,rgb.shape[0]//2), interpolation=cv2.INTER_AREA)
        small = cv2.resize(rgb, (32,24), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = bool(self.previous_small is not None and self.difference_exceeds(small, self.previous_small, 25))
        if cut:
            self.tracks = []
        scan = cut or not self.tracks or now-self.last_scan >= self.scan_interval
        if not scan:
            scan = self._track(rgb, gray) or not self.tracks
        if scan:
            self._scan(rgb, gray, now)
        def due(track):
            prediction = track["prediction"]
            if prediction is None:
                return True
            quiet_background = prediction["label"] == "background" and prediction["score"] >= .95
            interval = self.background_refresh_interval if quiet_background else self.refresh_interval
            return now-track["classified"] >= interval

        eligible = sorted((t for t in self.tracks if due(t)),key=lambda t:t["classified"])
        fresh_ids = set()
        for track in eligible[:self.budget]:
            x1,y1,x2,y2 = track["crop_box"]
            track["prediction"] = self.predictor.predict(rgb[y1:y2,x1:x2])
            track["classified"] = now
            fresh_ids.add(track["track_id"])
        elapsed = time.perf_counter()-start
        rows = []
        for track in self.tracks:
            prediction = track["prediction"]
            age = now+elapsed-track["classified"]
            if prediction is None or age > self.ttl:
                continue
            x1,y1,x2,y2 = track["box"]
            rows.append(dict(box=track["box"], crop_box=track["crop_box"], **prediction,
                       track_id=track["track_id"], classification_age_ms=age*1000,
                       classification_fresh=track["track_id"] in fresh_ids,
                       center_normalized=[(x1+x2)/rgb.shape[1]-1,(y1+y2)/rgb.shape[0]-1]))
        rows = suppress_duplicates(rows, "nested")
        self.previous_small, self.last_time = small, now
        self.frame_shape = rgb.shape
        return dict(observations=rows, processing_ms=(time.perf_counter()-start)*1000, full_scan=scan,
                    scene_cut=cut, inference_calls=len(fresh_ids), tracks=len(self.tracks),
                    target_lost=not any(r["accepted"] for r in rows),
                    flight_commands=None, deployment_approved=False)
