"""Preserve random streams across bounded training processes, not bitwise training replay."""

import random

import numpy as np


def capture_random_state(torch, loaders):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), mps=torch.mps.get_rng_state(),
                loaders={name: loader.generator.get_state() for name, loader in loaders.items()})


def restore_random_state(torch, loaders, state):
    if set(state["loaders"]) != set(loaders):
        raise ValueError("Random-state loader set differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.mps.set_rng_state(state["mps"])
    for name, loader in loaders.items():
        loader.generator.set_state(state["loaders"][name])


def seed_migrated_segment(torch, loaders, base_seed, start_epoch):
    """Explicit migration when an older checkpoint did not retain random states."""
    seed = base_seed + start_epoch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    for offset, loader in enumerate(loaders.values()):
        loader.generator.manual_seed(seed + offset)
    return seed
