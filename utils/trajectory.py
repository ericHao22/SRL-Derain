import numpy as np
import collections

Trajectory = collections.namedtuple('Trajectory', 'states reward_brisque')

def list_of_tuple_to_traj(l):
    states, reward_brisque, next_states =\
        [np.array(elem) for elem in zip(*l)]
    h, w = states.shape[-2:]
    states = np.concatenate([np.reshape(states[-1:], (1, -1, h, w)), next_states[-1:]], axis=1)
    traj = Trajectory(
        states = states.astype(np.float32), # [1, 3*2, h, w]
        reward_brisque = reward_brisque[-1:].astype(np.float32), # [1,]
    )
    
    return traj

def list_of_batch_tuple_to_traj(l):
    states, reward_brisque, next_states =\
        [np.array(elem) for elem in zip(*l)]
    N, B, C, h, w = states.shape
    states = np.concatenate([np.reshape(states[-1:], (B, C, h, w)), np.reshape(next_states[-1:], (B, C, h, w))], axis=1)
    traj = Trajectory(
        states = states.astype(np.float32), # [B, 3*2, h, w]
        reward_brisque = reward_brisque[-1].astype(np.float32), # [B,]
    )
    
    return traj