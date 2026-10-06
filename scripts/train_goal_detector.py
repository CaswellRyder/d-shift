"""Isolated offboard YOLO11n experiment; no viewer promotion or reserved-test evaluation."""

import argparse
from copy import copy, deepcopy
import os
from pathlib import Path
import shutil
import tempfile
import time

from dtr.data import read_json, sha256, write_json
from dtr.detector_nms import complete_cpu_nms
from dtr.detector_refinement import refinement_source, training_options
from dtr.training_checkpoint import capture_training_state, restore_training_state
from dtr.training_random import capture_random_state, restore_random_state, seed_migrated_segment
from dtr.training_resources import (DiskSpaceError, TrainingSegmentComplete, require_disk_space,
                                    require_segment_boundary, resume_stopper_state)

VALIDATION_REVISION = "blocking-transfer-complete-cpu-nms-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refine-from", help="Fresh lower-LR experiment from an immutable snapshot")
    parser.add_argument("--max-process-epochs", type=int, default=1,
                        help="Exit after this many saved epochs to bound native memory growth; 0 disables")
    args = parser.parse_args()
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    if args.epochs < 1 or args.batch < 1 or args.max_process_epochs < 0:
        parser.error("epochs and batch must be positive")
    if args.resume and args.refine_from:
        parser.error("Resume restores its recorded initialization; do not pass --refine-from")
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    storage_paths = [output, Path(tempfile.gettempdir())]
    require_disk_space(storage_paths)
    receipt = read_json(root / "receipt.json")
    if receipt["test_exported"] or set(receipt["splits"]) != {"train", "valid"}:
        raise ValueError("Only grouped train/valid data supported")
    if sha256(root / "dataset.yaml") != receipt["dataset_yaml_sha256"]:
        raise ValueError("Dataset YAML changed")
    # Keep settings local and disable external integrations before constructing models.
    config_dir = Path("artifacts/detector-settings").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO, settings
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.detect.val import DetectionValidator
    import torch
    import ultralytics

    settings.update({k: False for k in ("sync", "wandb", "mlflow", "clearml", "comet",
                                       "dvc", "hub", "neptune", "raytune", "tensorboard")})
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; choose a compute plan explicitly")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=args.resume)
    provenance = dict(dataset_receipt_sha256=sha256(root / "receipt.json"),
                      standard_sha256=sha256("configs/goal-detection-standard.json"),
                      torch=torch.__version__, ultralytics=ultralytics.__version__, device="mps",
                      test_evaluated=False, deployment_approved=False, epochs=args.epochs,
                      batch=args.batch, image_size=640, status="running", start_time=time.time())
    reset_ranking = False
    if args.resume:
        old = read_json(output / "experiment.json")
        for key in ("dataset_receipt_sha256", "standard_sha256", "torch", "ultralytics", "epochs", "batch"):
            if old[key] != provenance[key]:
                raise ValueError(f"Resume scope changed: {key}")
        if old["status"] == "complete":
            raise ValueError("Completed experiment cannot resume")
        model_path = output / "fit/weights/last.pt"
        reset_ranking = old.get("validation_revision") != VALIDATION_REVISION
        history = list(old.get("resume_history", []))
        history.append(dict(resumed_at=time.time(), previous_status=old["status"],
                            checkpoint_sha256=sha256(model_path),
                            previous_validation_revision=old.get("validation_revision"),
                            previous_random_state_resume=old.get("random_state_resume"),
                            previous_training_state_resume=old.get("training_state_resume"),
                            previous_max_process_epochs=old.get("max_process_epochs"),
                            ranking_reset=reset_ranking))
        provenance.update(start_time=old["start_time"], resume_history=history,
                          initial_model_sha256=old["initial_model_sha256"])
        provenance["refinement"] = old.get("refinement")
        expected_options = training_options(provenance["refinement"])
        if old.get("training_options", expected_options) != expected_options:
            raise ValueError("Recorded training options differ from this implementation")
        if reset_ranking:
            archive = output / f"before-validation-v2-{time.time_ns()}"
            archive.mkdir()
            for relative in ("experiment.json", "progress.json", "fit/results.csv",
                             "fit/weights/best.pt", "fit/weights/last.pt"):
                source = output / relative
                if source.is_file():
                    shutil.copy2(source, archive / source.name)
    else:
        provenance["refinement"] = (refinement_source(
            args.refine_from, receipt, "configs/goal-detection-standard.json"
        ) if args.refine_from else None)
        model_path = (Path(args.refine_from).resolve() / "model.pt" if args.refine_from
                      else Path("artifacts/pretrained/yolo11n.pt").resolve())
        if not model_path.is_file():
            raise FileNotFoundError("Download official yolo11n.pt to artifacts/pretrained first")
        provenance["initial_model_sha256"] = sha256(model_path)
        if (provenance["refinement"] and provenance["initial_model_sha256"]
                != provenance["refinement"]["model_sha256"]):
            raise ValueError("Refinement source changed during initialization")
    provenance["training_options"] = training_options(provenance["refinement"])
    provenance["validation_revision"] = VALIDATION_REVISION
    provenance["minimum_free_disk_gib"] = 10
    provenance["max_process_epochs"] = args.max_process_epochs
    provenance["process_policy"] = "Saved-boundary restarts; augmentation replay is not bitwise identical"
    write_json(output / "experiment.json", provenance)

    def restore_streams(trainer):
        actual_options = {key: getattr(trainer.args, key)
                          for key in provenance["training_options"]}
        if actual_options != provenance["training_options"]:
            raise ValueError("Trainer options differ from recorded profile")
        provenance["runtime_training_options_verified"] = True
        write_json(output / "experiment.json", provenance)
        if not args.resume:
            return
        loaders = dict(train=trainer.train_loader, valid=trainer.test_loader)
        state_path = output / "resume-state.json"
        saved = read_json(state_path) if state_path.exists() else {}
        if "random_state_file" in saved:
            path = output / saved["random_state_file"]
            if sha256(path) != saved["random_state_sha256"]:
                raise ValueError("Training random-state checksum mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["checkpoint_sha256"] != sha256(model_path):
                raise ValueError("Training random state belongs to another checkpoint")
            if "training" in payload:
                restore_training_state(trainer, payload["training"])
                provenance["training_state_resume"] = "full_precision_raw_model_and_optimizer"
                provenance["training_state_restoration_verified"] = True
            else:
                provenance["training_state_resume"] = "legacy_ema_model_fp16_optimizer"
            restore_random_state(torch, loaders, payload["state"])
            provenance["random_state_resume"] = dict(mode="restored", sha256=sha256(path))
        else:
            seed = seed_migrated_segment(torch, loaders, trainer.args.seed, trainer.start_epoch)
            provenance["random_state_resume"] = dict(mode="legacy_epoch_seed", seed=seed)
            provenance["training_state_resume"] = "legacy_ema_model_fp16_optimizer"
        write_json(output / "experiment.json", provenance)

    def progress(trainer):
        write_json(output / "progress.json", dict(
            epoch=trainer.epoch+1, epochs=trainer.epochs, metrics=trainer.metrics,
            updated_at=time.time(), test_evaluated=False, deployment_approved=False,
            validation_revision=VALIDATION_REVISION,
        ))
        if not trainer.stop:
            # Ultralytics saved last.pt before this callback. Exit before final_eval strips
            # optimizer state; persist patience so repeated resumes cannot extend early stopping.
            checkpoint_sha = sha256(output / "fit/weights/last.pt")
            random_path = output / f"resume-training-epoch-{trainer.epoch + 1}.pt"
            torch.save(dict(checkpoint_sha256=checkpoint_sha,
                            training=capture_training_state(torch, trainer),
                            state=capture_random_state(torch, dict(train=trainer.train_loader,
                                                                   valid=trainer.test_loader))), random_path)
            write_json(output / "resume-state.json", dict(
                checkpoint_sha256=checkpoint_sha,
                random_state_file=random_path.name, random_state_sha256=sha256(random_path),
                early_stopping=dict(best_fitness=float(trainer.stopper.best_fitness),
                                    best_epoch=int(trainer.stopper.best_epoch),
                                    possible_stop=bool(trainer.stopper.possible_stop)),
            ))
            require_segment_boundary(trainer.start_epoch, trainer.epoch,
                                     args.max_process_epochs, trainer.stop)

    class SafeValidator(DetectionValidator):
        def postprocess(self, preds):
            if self.args.task != "detect":
                raise ValueError("Offline adapter only supports plain detection")
            outputs = complete_cpu_nms(
                preds, conf_thres=self.args.conf, iou_thres=self.args.iou,
                multi_label=True, agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det, end2end=self.end2end,
            )
            outputs = [value.to(self.device, non_blocking=False) for value in outputs]
            return [dict(bboxes=x[:, :4], conf=x[:, 4], cls=x[:, 5], extra=x[:, 6:])
                    for x in outputs]

        def preprocess(self, batch):
            require_disk_space(storage_paths)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device, non_blocking=False)
            batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
            return batch

    class GuardedTrainer(DetectionTrainer):
        def resume_training(self, ckpt):
            super().resume_training(ckpt)
            if reset_ranking:
                self.best_fitness = 0.0
                self.stopper.best_fitness = 0.0
                self.stopper.best_epoch = self.start_epoch
            elif args.resume:
                state_path = output / "resume-state.json"
                state = resume_stopper_state(ckpt, read_json(state_path) if state_path.exists() else None,
                                             sha256(model_path))
                for key, value in state.items():
                    setattr(self.stopper, key, value)

        def get_validator(self):
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return SafeValidator(self.test_loader, save_dir=self.save_dir,
                                 args=copy(self.args), _callbacks=self.callbacks)

        def preprocess_batch(self, batch):
            # torch 2.6 MPS non-blocking copies can outlive overwritten CPU labels.
            # Blocking transfer retains correct labels in the reproduced failing batch.
            if self.device.type == "mps":
                if self.args.multi_scale:
                    raise ValueError("This MPS transfer adapter requires fixed image size")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device, non_blocking=False)
                batch["img"] = batch["img"].float() / 255
            else:
                batch = super().preprocess_batch(batch)
            boxes = batch["bboxes"]
            if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
                raise ValueError("Corrupt normalized training boxes after device transfer")
            self.has_targets = bool(batch["cls"].numel())
            if (self.has_targets and not hasattr(self, "captured_initial_batch")
                    and not (output / "initial-batch.pt").exists()):
                self.captured_initial_batch = True
                torch.save(dict(model=deepcopy(self.model).cpu(), batch={
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }), output / "initial-batch.pt")
            return batch

    def guard(trainer):
        require_disk_space(storage_paths)
        if trainer.has_targets and float(trainer.loss_items[0]) == 0:
            trainer.empty_box_batches = getattr(trainer, "empty_box_batches", 0) + 1
        elif trainer.has_targets:
            trainer.empty_box_batches = 0
        if getattr(trainer, "empty_box_batches", 0) >= 10:
            raise RuntimeError("No box-regression signal in 10 consecutive labeled batches")

    try:
        model = YOLO(str(model_path))
        if provenance["refinement"] and not args.resume:
            if ([model.names[i] for i in range(len(model.names))] != receipt["classes"]
                    or sha256(model_path) != provenance["initial_model_sha256"]):
                raise ValueError("Loaded refinement model class order or checksum differs")
        model.add_callback("on_train_start", restore_streams)
        model.add_callback("on_fit_epoch_end", progress)
        model.add_callback("on_train_batch_end", guard)
        if args.resume:
            model.train(resume=True, device="mps", trainer=GuardedTrainer)
        else:
            model.train(trainer=GuardedTrainer, data=str(root / "dataset.yaml"), project=str(output), name="fit",
                        epochs=args.epochs, batch=args.batch, imgsz=640, device="mps", workers=0,
                        seed=42, deterministic=True, patience=8, cache=False, amp=False,
                        optimizer="AdamW", **provenance["training_options"],
                        hsv_h=0.0, hsv_s=0.3, hsv_v=0.2, fliplr=0.5, flipud=0.0,
                        degrees=5.0, scale=0.3,
                        plots=False, save=True, save_period=-1, val=True, verbose=False,
                        resume=False)
        provenance.update(status="complete", finished_at=time.time(),
                          best_model_sha256=sha256(output / "fit/weights/best.pt"))
    except TrainingSegmentComplete as error:
        provenance.update(status="interrupted", interruption_reason="process_epoch_limit",
                          error=str(error), finished_at=time.time())
        print(str(error), flush=True)
        return 75
    except BaseException as error:
        provenance.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, DiskSpaceError)) else "failed",
                          error=str(error), finished_at=time.time())
        raise
    finally:
        write_json(output / "experiment.json", provenance)


if __name__ == "__main__":
    raise SystemExit(main())
