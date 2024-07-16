import argparse
import numpy as np
import os
from tqdm import tqdm
from brisque import BRISQUE
from mini_batch_loader import MiniBatchLoader
from MyFCN import *
import torch
import torch.optim as optim
from utils.traj_dataset import TrajectoryDataset
from utils.trajectory import list_of_tuple_to_traj
from utils.compute_Rbrisque import brisque_reward
from utils.augmentation import aug_mask

def compute_diff(image1, image2):
    return np.mean(np.abs(image1 - image2))

def main(args):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        args.data_path, 
        args.image_dir_path)
    
    brisque_metrics = BRISQUE(url=False)

    train_data_size = MiniBatchLoader.count_paths(args.data_path)

    # criterion for training Rnet
    CE = torch.nn.CrossEntropyLoss()
    
    # for saving trajectories
    trajectory_dataset = TrajectoryDataset()
    # init Reward function
    r_net_brisque = Reward_Predictor(image_size=(args.img_size, args.img_size)).cuda()
    optimizer_rnet_brisque = optim.Adam(r_net_brisque.parameters(), lr=args.lr)
    
    # ============= pretraining Rnet with augmented mask ==============
    n = 20
    for i in tqdm(range(0, train_data_size)):
        raw_x, pseudo_ys, mask, name = mini_batch_loader.load_training_data(index=i)
        if np.sum(mask) == 0: continue
        original_rainy_image = raw_x[0].transpose([1, 2, 0]) # (h, w, 3)
        rain_mask = mask[0][0] # (h, w)
        # ============= generate aug images ==============
        aug_rain_image_list = []
        for _ in range(n):
            # random percentage of rain add to image: [0.0, 0.6]
            p_rain = np.random.random() * (0.6-0.0) + 0.0
            random_aug_mask = aug_mask(original_rainy_image, rain_mask, p_rain)
            # random intensity of rain add to image
            p_intensity = np.random.rand(rain_mask.shape[0], rain_mask.shape[1]) * 0.5
            aug_rain_image = np.zeros_like(original_rainy_image).astype(np.float32)
            for i in range(3):
                aug_rain_image[:, :, i] = np.where(random_aug_mask > 0, original_rainy_image[:, :, i] + p_intensity * random_aug_mask, original_rainy_image[:, :, i])
            aug_rain_image = np.clip(aug_rain_image, 0, 1)
            aug_rain_image_list.append(aug_rain_image)

        # ============= gererate trajectories ==============
        for i in range(n):
            for j in range(i+1, n):
                # transition_tuple: (state_t[3, h, w], reward_brisque[1], state_t+1[3, h, w])
                s1 = aug_rain_image_list[i].transpose([2, 0, 1])
                s2 = aug_rain_image_list[j].transpose([2, 0, 1])
                h, w = s1.shape[-2:]
                # random crop s1, s2 to args.img_size
                rand_range_h = h-args.img_size
                rand_range_w = w-args.img_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                s1 = s1[:, y_offset:y_offset+args.img_size, x_offset:x_offset+args.img_size]
                s2 = s2[:, y_offset:y_offset+args.img_size, x_offset:x_offset+args.img_size]
                diff = compute_diff(s1, s2)
                if diff * 255 < 20:
                    continue
                # s1 --> s2
                reward_brisque = brisque_reward(brisque_metrics, s1, s2)
                transition_tuple = (s1, reward_brisque, s2)    
                traj = list_of_tuple_to_traj([transition_tuple])
                trajectory_dataset.add_traj(traj)
                # s2 --> s1
                reward_brisque = brisque_reward(brisque_metrics, s2, s1)
                transition_tuple = (s2, reward_brisque, s1)
                traj = list_of_tuple_to_traj([transition_tuple])
                trajectory_dataset.add_traj(traj)

    print("total {} trajectories.".format(trajectory_dataset.len()))
    
    # ============= pre-training Rnet ==============
    ru = 0
    for _ in tqdm(range(args.N_pre)):
        ru += 1
        x1, x2, R1_b, R2_b = trajectory_dataset.batch_sample(batch_size=args.batch_size)
        x1 = torch.from_numpy(x1).cuda()
        x2 = torch.from_numpy(x2).cuda()
        R1_b = torch.tensor(R1_b).cuda()
        R2_b = torch.tensor(R2_b).cuda()
        y_brisque = torch.where(R1_b < R2_b, 1, 0)
        y_brisque = y_brisque.to(dtype=torch.long)
        v1_b = r_net_brisque(x1)
        v2_b = r_net_brisque(x2)
        logits_b = torch.cat((v1_b, v2_b), dim=1)
        rnet_brisque_loss = CE(logits_b, y_brisque.squeeze(-1))
        optimizer_rnet_brisque.zero_grad()
        rnet_brisque_loss.backward()
        optimizer_rnet_brisque.step()
    
    # save Rnet model
    os.makedirs(os.path.join(args.save_dir_path, 'model_weight'), exist_ok=True)
    torch.save(r_net_brisque.state_dict(), os.path.join(args.save_dir_path, 'model_weight', 'rnet_brisque.pt'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for training') 
    # Directories
    parser.add_argument('--image_dir_path', type=str, default='dataset/')
    parser.add_argument('--data_path', type=str, default='dataset/Rain100L/testing.txt')
    parser.add_argument('--save_dir_path', type=str, default='./Results/Rain100L/test/Rnet+/')
    # config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--N_pre', type=int, default=6000)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    main(args)