"""One-time, CPU-only import of public timm weights; training remains native Keras."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .models import CHECKPOINT, MEAN, STD, teacher


def import_weights(output, size=96):
    import torch
    import timm
    import keras

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    torch.set_num_threads(2)
    reference = timm.create_model(CHECKPOINT, pretrained=True).cpu().eval()
    model = teacher(1000, size)
    modules = dict(reference.named_modules())
    mapped = []
    for layer in model.layers:
        if not layer.weights:
            continue
        key = layer.name.replace("__", ".")
        source = modules[key]
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            values = [source.weight.detach().numpy().transpose(2, 3, 0, 1)]
        elif isinstance(layer, keras.layers.Conv2D):
            values = [source.weight.detach().numpy().transpose(2, 3, 1, 0)]
        elif isinstance(layer, keras.layers.BatchNormalization):
            if abs(source.eps - layer.epsilon) > 1e-12:
                raise ValueError(f"BN epsilon mismatch: {key}")
            values = [
                v.detach().numpy()
                for v in (source.weight, source.bias, source.running_mean, source.running_var)
            ]
        elif isinstance(layer, keras.layers.Dense):
            values = [source.weight.detach().numpy().T, source.bias.detach().numpy()]
        else:
            raise TypeError(f"Unmapped layer {key}")
        layer.set_weights(values)
        mapped.append(key)
    # Assert every source parameter was consumed, not just shape-compatible Keras layers.
    missing = [
        name for name, _ in reference.named_parameters() if name.rsplit(".", 1)[0] not in mapped
    ]
    if missing:
        raise ValueError(f"Unmapped source parameters: {missing}")
    rng = np.random.default_rng(42)
    samples = rng.uniform(0, 255, (3, size, size, 3)).astype(np.float32)
    normalized = ((samples / 255 - MEAN) / STD).astype(np.float32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(normalized.transpose(0, 3, 1, 2))).numpy()
    actual = model(samples, training=False).numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    receipt = {
        "checkpoint": CHECKPOINT,
        "source": "timm / Ross Wightman ImageNet-1k",
        "license": "Apache-2.0",
        "size": size,
        "mapped_layers": len(mapped),
        "max_abs_logit_error": float(np.max(np.abs(actual - expected))),
        "keras_parameters": model.count_params(),
        "parity_passed": True,
        "dtr_trained": False,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
