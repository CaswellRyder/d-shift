"""Full-precision training continuation, separate from the EMA inference checkpoint."""

FORMAT = "full-precision-training-v1"


def cpu_copy(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", non_blocking=False).clone()
    if isinstance(value, dict):
        return {key: cpu_copy(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(torch, item) for item in value)
    return value


def capture_training_state(torch, trainer):
    return cpu_copy(torch, dict(
        format=FORMAT, model=trainer.model.state_dict(), ema=trainer.ema.ema.state_dict(),
        ema_updates=trainer.ema.updates, optimizer=trainer.optimizer.state_dict(),
        scheduler=trainer.scheduler.state_dict(), scaler=trainer.scaler.state_dict(),
    ))


def same_state(torch, expected, actual):
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and expected.dtype == actual.dtype
                and torch.equal(expected, actual))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(same_state(torch, value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (type(expected) is type(actual) and len(expected) == len(actual)
                and all(same_state(torch, a, b) for a, b in zip(expected, actual)))
    return expected == actual


def restore_training_state(trainer, state):
    import torch
    if state["format"] != FORMAT:
        raise ValueError("Unsupported full-precision training state")
    # Copy into existing parameters, preserving optimizer parameter identities.
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.ema.ema.load_state_dict(state["ema"], strict=True)
    trainer.ema.updates = state["ema_updates"]
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.scaler.load_state_dict(state["scaler"])
    if not same_state(torch, state, capture_training_state(torch, trainer)):
        raise ValueError("Full-precision training state differs after restoration")
