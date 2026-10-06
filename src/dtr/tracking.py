"""Display-only greedy IoU association. Not predictive or flight-qualified tracking."""

from collections import deque


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class BoxTracker:
    def __init__(self, threshold=0.25, ttl=0.75):
        self.threshold, self.ttl = threshold, ttl
        self.tracks = {}
        self.next_id = 1

    def update(self, observations, now):
        self.tracks = {k: v for k, v in self.tracks.items() if now - v["last"] <= self.ttl}
        pairs = sorted(
            (
                (iou(row["box"], t["box"]), i, k)
                for i, row in enumerate(observations)
                for k, t in self.tracks.items()
            ),
            reverse=True,
        )
        matches, used = {}, set()
        for overlap, i, k in pairs:
            if overlap >= self.threshold and i not in matches and k not in used:
                matches[i] = k
                used.add(k)
        result = []
        for i, row in enumerate(observations):
            k = matches.get(i)
            if k is None:
                k = self.next_id
                self.next_id += 1
                self.tracks[k] = {"trail": deque(maxlen=20)}
            track = self.tracks[k]
            x1, y1, x2, y2 = row["box"]
            track.update(box=row["box"], last=now)
            track["trail"].append([(x1 + x2) / 2, (y1 + y2) / 2])
            result.append({**row, "track_id": k, "trail": list(track["trail"])})
        return result
