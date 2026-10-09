from scripts.pi_pixel_balloon_bench import boxes_to_detections, min_size
from scripts.review_pixel_balloon_bench import panel_counts


def test_min_shape_size_keeps_demo_image_fraction():
    assert min_size(540, 540) == 200
    assert min_size(320, 240) == 53
    assert min_size(640, 480) == 211


def test_inclusive_pixel_bounds_become_half_open_boxes_ordered_by_area():
    rows = boxes_to_detections([(10, 20, 19, 29), (0, 0, 1, 1)], "red_balloon")
    assert rows[0] == dict(label="red_balloon", box=[10, 20, 20, 30], score=100.0, accepted=True)
    assert rows[1]["score"] == 4.0


def test_panel_counts_charge_wrong_color_and_duplicates():
    frames = [
        dict(panel="indoor", truth=[dict(label="red_balloon", box=[0, 0, 10, 10])]),
        dict(panel="original", truth=[dict(label="blue_balloon", box=[0, 0, 10, 10])]),
    ]
    detections = [
        boxes_to_detections([(0, 0, 9, 9), (0, 0, 8, 8)], "red_balloon"),
        boxes_to_detections([(0, 0, 9, 9)], "red_balloon"),
    ]
    counts = panel_counts(frames, detections)
    assert counts["indoor"]["red_balloon"] == dict(tp=1, fp=1, fn=0)
    assert counts["original"]["red_balloon"] == dict(tp=0, fp=1, fn=0)
    assert counts["original"]["blue_balloon"] == dict(tp=0, fp=0, fn=1)
