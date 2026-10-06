"""Native Keras implementation of MobileNetV4-Conv-Small, width 1.0.

Block topology follows Google Model Garden and timm 1.0.15 (Apache-2.0).
Symmetric padding and BN epsilon match the selected timm checkpoint exactly.
All models consume NHWC RGB float32 values in [0, 255]; normalization is embedded.
"""

import keras
import numpy as np
from keras import layers as L

CHECKPOINT = "mobilenetv4_conv_small.e2400_r224_in1k"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def lname(name):
    return name.replace(".", "__")


def conv(x, channels, kernel, stride, name, depthwise=False):
    if kernel > 1:
        x = L.ZeroPadding2D(kernel // 2, name=lname(name + ".pad"))(x)
    opts = dict(
        kernel_size=kernel, strides=stride, padding="valid", use_bias=False, name=lname(name)
    )
    if depthwise:
        return L.DepthwiseConv2D(**opts)(x)
    return L.Conv2D(channels, **opts)(x)


def bn(x, name, activation=True):
    x = L.BatchNormalization(epsilon=1e-5, momentum=0.9, name=lname(name))(x)
    return L.ReLU(name=lname(name + ".act"))(x) if activation else x


def cna(x, channels, kernel, stride, name, depthwise=False, activation=True):
    x = conv(x, channels, kernel, stride, name + ".conv", depthwise)
    return bn(x, name + ".bn", activation)


def uib(x, channels, start, middle, stride, expansion, name):
    shortcut = x
    incoming = int(x.shape[-1])
    if start:
        x = cna(
            x,
            incoming,
            start,
            stride if not middle else 1,
            name + ".dw_start",
            depthwise=True,
            activation=False,
        )
    expanded = int(incoming * expansion + 4) // 8 * 8
    x = cna(x, expanded, 1, 1, name + ".pw_exp")
    if middle:
        x = cna(x, expanded, middle, stride, name + ".dw_mid", depthwise=True)
    x = cna(x, channels, 1, 1, name + ".pw_proj", activation=False)
    if stride == 1 and incoming == channels:
        x = L.Add(name=lname(name + ".residual"))([shortcut, x])
    return x


def teacher(num_classes=1000, size=96):
    if size < 32 or num_classes < 2:
        raise ValueError("Teacher requires size >= 32 and at least two classes")
    inputs = keras.Input((size, size, 3), name="rgb_0_255")
    x = L.Rescaling(
        scale=[1 / (255 * s) for s in STD],
        offset=[-m / s for m, s in zip(MEAN, STD)],
        name="imagenet_normalization",
    )(inputs)
    x = bn(conv(x, 32, 3, 2, "conv_stem"), "bn1")
    for stage, specs in enumerate([[(32, 3, 2), (32, 1, 1)], [(96, 3, 2), (64, 1, 1)]]):
        for block, (ch, k, stride) in enumerate(specs):
            p = f"blocks.{stage}.{block}"
            x = bn(conv(x, ch, k, stride, p + ".conv"), p + ".bn1")
    for stage, specs in [
        (2, [(96, 5, 5, 2, 3)] + [(96, 0, 3, 1, 2)] * 4 + [(96, 3, 0, 1, 4)]),
        (
            3,
            [
                (128, 3, 3, 2, 6),
                (128, 5, 5, 1, 4),
                (128, 0, 5, 1, 4),
                (128, 0, 5, 1, 3),
                (128, 0, 3, 1, 4),
                (128, 0, 3, 1, 4),
            ],
        ),
    ]:
        for block, args in enumerate(specs):
            x = uib(x, *args, name=f"blocks.{stage}.{block}")
    x = bn(conv(x, 960, 1, 1, "blocks.4.0.conv"), "blocks.4.0.bn1")
    x = L.GlobalAveragePooling2D(keepdims=True, name="global_pool")(x)
    x = bn(conv(x, 1280, 1, 1, "conv_head"), "norm_head")
    x = L.Flatten(name="flatten")(x)
    logits = L.Dense(num_classes, name="classifier")(x)
    return keras.Model(inputs, logits, name="mobilenetv4_conv_small")


def student(num_classes, size=64):
    inputs = keras.Input((size, size, 3), name="rgb_0_255")
    x = L.Rescaling(1 / 255.0, name="normalize")(inputs)
    for channels in (8, 16, 32):
        x = L.Conv2D(
            channels, 3, strides=2, padding="same", activation="relu", name=f"conv_{channels}"
        )(x)
    x = L.GlobalAveragePooling2D(name="pool")(x)
    return keras.Model(inputs, L.Dense(num_classes, name="classifier")(x), name="tiny_student")


def research_student(num_classes, size=64, variant="tiny"):
    """Controlled crop ablations; not a new MobileNet or a deployment default.

    spatial: preserve a 2x2 layout rather than average all locations together.
    context: also add a 3x3 convolution at stride eight (RF 15 -> 31 pixels).
    separable_context: same spatial extent with depthwise + pointwise filtering.
    All retain the original input contract and inexpensive three-layer stem.
    """
    if variant not in ("tiny", "spatial", "context", "separable_context"):
        raise ValueError("Unknown research student variant")
    if size < 32 or size % 16 or num_classes < 2:
        raise ValueError("Require size >=32 divisible by 16 and at least two classes")
    if variant == "tiny":
        return student(num_classes, size)
    inputs = keras.Input((size, size, 3), name="rgb_0_255")
    x = L.Rescaling(1 / 255.0, name="normalize")(inputs)
    for channels in (8, 16, 32):
        x = L.Conv2D(channels, 3, strides=2, padding="same", activation="relu",
                     name=f"conv_{channels}")(x)
    if variant == "context":
        x = L.Conv2D(32, 3, padding="same", activation="relu", name="context")(x)
    elif variant == "separable_context":
        x = L.SeparableConv2D(32, 3, padding="same", activation="relu",
                              name="separable_context")(x)
    x = L.AveragePooling2D(pool_size=size // 16, name="spatial_pool")(x)
    x = L.Flatten(name="spatial_features")(x)
    return keras.Model(inputs, L.Dense(num_classes, name="classifier")(x),
                       name=f"{variant}_student")


def initialize_separable_context(source, target):
    """Rank-one per-input-channel approximation, followed by separately scored fine-tuning.

    Copy the trained stem/head exactly. SVD is a weight approximation, not a claim
    of equivalent activations or preserved accuracy. Never mutates the source.
    """
    if source.input_shape != target.input_shape or source.output_shape != target.output_shape:
        raise ValueError("Warm-start input/output contract mismatch")
    kernel, bias = source.get_layer("context").get_weights()
    layer = target.get_layer("separable_context")
    depthwise, pointwise, target_bias = layer.get_weights()
    if (kernel.shape != (3, 3, 32, 32) or depthwise.shape != (3, 3, 32, 1)
            or pointwise.shape != (1, 1, 32, 32) or target_bias.shape != bias.shape):
        raise ValueError("Require the matched 32-channel context architectures")
    approximation = np.empty_like(kernel)
    for channel in range(32):
        u, values, vh = np.linalg.svd(kernel[:, :, channel, :].reshape(9, 32), full_matrices=False)
        a, b = u[:, 0]*np.sqrt(values[0]), vh[0]*np.sqrt(values[0])
        depthwise[:, :, channel, 0] = a.reshape(3, 3)
        pointwise[0, 0, channel, :] = b
        approximation[:, :, channel, :] = np.outer(a, b).reshape(3, 3, 32)
    for name in ("conv_8", "conv_16", "conv_32", "classifier"):
        target.get_layer(name).set_weights(source.get_layer(name).get_weights())
    layer.set_weights([depthwise, pointwise, bias])
    return dict(method="per_input_channel_rank_one_svd", copied_layers=["conv_8", "conv_16", "conv_32", "classifier"],
                relative_kernel_error=float(np.linalg.norm(kernel-approximation) / max(np.linalg.norm(kernel), 1e-12)))


def load_backbone(model, pretrained):
    """Copy only identically named/shaped layers; never import ImageNet classifier."""
    for layer in model.layers:
        if layer.weights and layer.name != "classifier":
            layer.set_weights(pretrained.get_layer(layer.name).get_weights())


def freeze_backbone(model, frozen):
    for layer in model.layers:
        # Preserve pretrained running statistics on a small DTR dataset.
        layer.trainable = layer.name == "classifier" or (
            not frozen and not isinstance(layer, L.BatchNormalization)
        )
