import numpy as np

def identity(state):
    return state

def polynomial(state):
    if state.n_dim == 1:
        return np.array([(np.asscalar(state) / 20) ** i for i in range(6)])
    elif state.n_dim == 3:
        return np.concatenate([(state / 20) ** i for i in range(6)], axis=2)
    else:
        assert False