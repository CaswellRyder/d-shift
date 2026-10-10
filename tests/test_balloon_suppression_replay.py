from dtr.vision import suppress_duplicates
from scripts.evaluate_red_blue_development import detection_counts
from scripts.replay_balloon_suppression import raw, replay
from scripts.research_balloon_search import suppress_confirmed_balloon_parts


def test_replay_reproduces_saved_counts_then_splits_merged_parent():
    found = [dict(box=[42, 163, 175, 233], label="red_balloon", accepted=True, score=.99),
             dict(box=[43, 166, 104, 225], label="red_balloon", accepted=True, score=1.),
             dict(box=[113, 177, 169, 232], label="red_balloon", accepted=True, score=1.),
             dict(box=[0, 0, 30, 30], label="background", accepted=True, score=.9)]
    truth = [dict(label="red_balloon", box=[40.5, 165.5, 100, 225.7]),
             dict(label="red_balloon", box=[114.4, 175.6, 170.1, 232.4])]
    saved = suppress_confirmed_balloon_parts(suppress_duplicates(found))
    frame = dict(id=11, truth=truth, detections=saved, counts=detection_counts(truth, saved))
    assert [d["accepted"] for d in raw(saved)] == [True, True, True, True]
    current, drift = replay([frame], split=False)
    assert not drift and current["red_balloon"] == dict(tp=1, fp=1, fn=1)
    split, changed = replay([frame], split=True)
    assert split["red_balloon"] == dict(tp=2, fp=0, fn=0) and changed[0]["frame"] == 11
