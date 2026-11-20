"""Teacher-side strata. Never supplied as encoder or score-head input."""
import numpy as np

GROUP_NAMES = ['building', 'stationary_objects', 'disconnected_structure', 'registration', 'dynamic']


def loss_groups(attributes):
    role = attributes['source_role']
    group = np.zeros(len(role), np.uint8)
    group[role == 5] = 1
    group[attributes['is_disconnected_control']] = 2
    group[np.isin(role, [2, 6])] = 3
    group[role == 7] = 4
    return group
