"""Create a separate train-exposure experiment; validation and reserved test stay untouched."""

import argparse

from dtr.detector_sampling import prepare_close_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--copies", type=int, default=4)
    args = parser.parse_args()
    result = prepare_close_range(args.source, args.output, threshold=args.threshold, copies=args.copies)
    print(dict(frames={s: r["frames"] for s, r in result["splits"].items()},
               selected=len(result["training_exposure"]["selected_files"]),
               qualifying_boxes=result["training_exposure"]["qualifying_boxes_by_class"],
               test_evaluated=False, deployment_approved=False))


if __name__ == "__main__":
    main()
