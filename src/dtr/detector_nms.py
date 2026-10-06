"""Offline detector postprocessing: complete every batch, without a silent time cutoff.

Optional detector dependencies are imported only after the caller sets local/offline settings.
This is an accuracy-evaluation adapter, not a bounded-latency flight runtime.
"""


def complete_cpu_nms(predictions, **kwargs):
    import torch
    from ultralytics.utils.nms import non_max_suppression

    raw = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    # NMS modifies coordinates in place; keep the model output intact for validation loss.
    raw = raw.detach().to(device="cpu", dtype=torch.float32).clone()
    return non_max_suppression(raw, max_time_img=float("inf"), **kwargs)


def offline_predictor():
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops

    class CompletePredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            if self.args.task != "detect" or getattr(self, "_feats", None) is not None:
                raise ValueError("Offline adapter only supports plain detection")
            outputs = complete_cpu_nms(
                preds, conf_thres=self.args.conf, iou_thres=self.args.iou,
                classes=self.args.classes, agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det, end2end=getattr(self.model, "end2end", False),
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            return self.construct_results(outputs, img, orig_imgs, **kwargs)

    return CompletePredictor
