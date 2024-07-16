import numpy as np


class TrajectoryDataset(object):
    """
    Given a set of trajectory, provide a preference between every possible pair of trajectories
    """

    def __init__(self):
        self.trajs = []

    def add_traj(self, new_traj):
        self.trajs.append(new_traj)

    def len(self):
        return len(self.trajs)
    
    def reset_trajs(self):
        self.trajs = []
    
    def batch_sample(self, batch_size):
        #########################
        # Return format
        # tau_is, tau_js: [B, c, h, w]
        # R1_bs, R2_bs: [B, 1]
        def sample():
            N = len(self.trajs)
            a, b = np.random.choice(N,2,replace=False)
            x1, x2 = self.trajs[a].states, self.trajs[b].states
            R1_b, R2_b = np.sum(self.trajs[a].reward_brisque), np.sum(self.trajs[b].reward_brisque)
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
