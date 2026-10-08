"""Measure fit on admitted TRAIN crops only; explicitly not detector generalization."""
import argparse
from pathlib import Path

from dtr.data import sha256, write_json
from dtr.runtime import Predictor
from dtr.vision import RED_BLUE_PROFILE, validate_model_profile
from scripts.build_indoor_balloon_training import admitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue",required=True)
    p.add_argument("--review",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--model",action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = admitted(args.queue,args.review,args.manifest)
    results = {}
    for item in args.model:
        name,path = item.split("=",1)
        if name in results:
            raise ValueError("Duplicate model name")
        predictor = Predictor(path,allow_unvalidated=True)
        validate_model_profile(predictor.metadata,RED_BLUE_PROFILE)
        predictions = []
        for row in rows:
            pred = predictor.predict(Path(args.queue)/row["path"])
            predictions.append(dict(id=row["id"],truth=row["label"],label=pred["label"],
                                    accepted=pred["accepted"],score=pred["score"]))
        summary = {}
        for label in predictor.metadata["classes"]:
            group = [r for r in predictions if r["truth"] == label]
            summary[label] = dict(total=len(group),
                correct_argmax=sum(r["label"] == label for r in group),
                correct_accepted=sum(r["accepted"] and r["label"] == label for r in group),
                accepted_as_balloon=sum(r["accepted"] for r in group))
        results[name] = dict(model_sha256=sha256(path),threshold=predictor.metadata["threshold"],
                             summary=summary,predictions=predictions)
        print(name,summary,flush=True)
    write_json(args.output,dict(scope="Fit on 78 admitted training crops, NOT generalization or full-frame metrics",
        queue_sha256=sha256(Path(args.queue)/"review.json"),review_sha256=sha256(args.review),
        script_sha256=sha256(__file__),results=results,test_evaluated=False,deployment_approved=False))


if __name__ == "__main__":
    main()
