import numpy as np

def identity(state, mask = None):
    return state

def polynomial(state, mask = None):
    if state.ndim == 1:
        return np.array([np.asscalar(state)**i / 20**i if state != 0 else 0 for i in range(6)])
    elif state.ndim == 3:
        return np.concatenate([(1 - mask)[:, :, np.newaxis] * (state ** i) / (20**i) for i in range(6)], axis=2)

    else:
        assert False
