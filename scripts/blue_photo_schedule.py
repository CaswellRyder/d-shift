"""Fixed decay independent of metrics, panel results and checkpoint selection."""
def learning_rate(protocol):
    schedule=protocol['learning_rate_schedule']
    if schedule!={'kind':'cosine','decay_steps':26640,'alpha':.1} or protocol['learning_rate']!=.0001:
        raise ValueError('Require fixed26640-step cosine decay from1e-4 to1e-5')
    import keras
    return keras.optimizers.schedules.CosineDecay(protocol['learning_rate'],schedule['decay_steps'],alpha=schedule['alpha'])
