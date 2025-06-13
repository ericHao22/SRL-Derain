import numpy as np


class TrajectoryDataset(object):
    """
    Given a set of trajectory, provide a preference between every possible relative pair of trajectories
    """

    def __init__(self):
        self.traj_pairs = []
    
    def add_traj_pair(self, forward_traj, backward_traj):
        self.traj_pairs.append([forward_traj, backward_traj])

    def len(self):
        return len(self.traj_pairs) * 2
    
    def reset(self):
        self.traj_pairs = []
    
    def batch_sample(self, batch_size):
        #########################
        # Return format
        # tau_is, tau_js: [B, c, h, w]
        # R1_bs, R2_bs: [B, 1]
        def sample():
            N = len(self.traj_pairs)
            index = np.random.choice(N, 1, replace=False)[0]
            forward, backward = self.traj_pairs[index]
            x1, x2 = forward.states, backward.states
            R1_b, R2_b = np.sum(forward.reward_brisque), np.sum(backward.reward_brisque)
            return x1, x2, R1_b, R2_b

        tau_is = []; tau_js = []; R1_bs = []; R2_bs = []
        for _ in range(batch_size):
            tau_i, tau_j, R1_b, R2_b = sample()
            tau_is.append(tau_i[0])
            tau_js.append(tau_j[0])
            R1_bs.append([R1_b])
            R2_bs.append([R2_b])
        
        tau_is = np.asarray(tau_is)
        tau_js = np.asarray(tau_js)
        R1_bs = np.asarray(R1_bs)
        R2_bs = np.asarray(R2_bs)
        
        return tau_is, tau_js, R1_bs, R2_bs
