"""Reduce matched spatial-head evidence, checking model and evaluation identities."""
import argparse
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.summarize_public_balloon_trial import checked_metrics


def summarize():
    config='configs/balloon-spatial-box-20261011.json'
    audit_path='runs/spatial-box-grouped-audit-20261011.json'
    audit=read_json(audit_path)
    models={}
    partitions=[]
    for grid in (2,4):
        run=Path(f'runs/balloon-spatial-box-grid{grid}-20261011')
        model=run/'box.fp32.tflite'
        meta=read_json(model.with_suffix('.json'))
        report=read_json(run/'report.json')
        if (meta['sha256']!=sha256(model) or meta['pool_grid']!=grid
                or meta['config_sha256']!=sha256(config)
                or meta['source_tflite_sha256']!=audit['parent_model_sha256']
                or meta['source_student_sha256']!=audit['parent_checkpoint_sha256']
                or meta['script_sha256']!=audit['training_script_sha256']
                or report['max_keras_tflite_delta']>.002
                or meta['deployment_approved'] is not False):
            raise ValueError('Matched box model identity changed')
        partitions.append(report['source_partition'])
        models[str(grid)]=dict(path=str(model),metadata=meta,
            report_sha256=sha256(run/'report.json'),model_sha256=sha256(model),
            max_keras_tflite_delta=report['max_keras_tflite_delta'],
            fit={k:{a:b for a,b in report[k].items() if a!='examples'} for k in ('training','label_holdout')})
    if partitions[0]!=partitions[1]:
        raise ValueError('Head comparison uses different source partitions')
    panels={}
    for panel in ('engdes2-lockbox-panel3-20261010','engdes2-development-panel2-20261010',
                  'engdes2-development-scenes-20261008'):
        comparison={}
        for grid in (2,4):
            path=Path(f'runs/spatial-box-grid{grid}-{panel}-20261011.json')
            result=read_json(path)
            if (result['model_sha256']!=models[str(grid)]['model_sha256']
                    or result['threshold']!=.8 or result['matching_iou']!=.5
                    or result['max_class_probability_delta']>1e-5
                    or result['test_evaluated'] is not False
                    or result['deployment_approved'] is not False):
                raise ValueError('Scene comparison changed')
            comparison[str(grid)]=dict(path=str(path),sha256=sha256(path),
                queue_sha256=result['queue_sha256'],review_sha256=result['review_sha256'],
                max_class_probability_delta=result['max_class_probability_delta'],
                results={k:checked_metrics(v) for k,v in result['results'].items()})
        for key in ('queue_sha256','review_sha256'):
            if comparison['2'][key]!=comparison['4'][key]:
                raise ValueError('Different scene reviews across heads')
        if comparison['2']['results']['parent']!=comparison['4']['results']['parent']:
            raise ValueError('Different parent baseline')
        panels[panel]=comparison
    return dict(models=models,panels=panels,
        grouped_audit=dict(path=audit_path,sha256=sha256(audit_path),
            results={k:dict(metrics={a:b for a,b in v['metrics'].items() if a!='examples'},
                            folds=v['folds']) for k,v in audit['results'].items()},
            scope=audit['scope']),
        script_sha256=sha256(__file__),
        decision='Neither head promoted: red localization regresses and blue gains are inconsistent',
        scope='Repeatedly inspected development panels, not independent qualification. No new classifier training or Pi changes.',
        test_evaluated=False,deployment_approved=False,pi_tested=False,qualification_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize())
