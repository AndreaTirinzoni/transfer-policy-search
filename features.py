import numpy as np

def identity(state, mask = None):
    """
    This function returns the identity of the state
    :param state: The states of all the episodes in all time steps
    :param mask: A mask associated to the episodes length
    :return: A matrix representing the identity of the states
    """
    return state

def polynomial(state, mask = None):
    """
    This function returns the features of order 4 of the state
    :param state: The states of all the episodes in all time steps
    :param mask: A mask associated to the episodes length
    :return: A matrix representing the new features
    """
    if state.ndim == 1:
        return np.array([np.asscalar(state)**i / 20**i if state != 0 else 0 for i in range(4)])
    elif state.ndim == 3:
        return np.concatenate([(np.ones(mask.shape) - mask)[:, :, np.newaxis] * (state ** i) / (20**i) for i in range(4)], axis=2)

    else:
        assert False
