import argparse
import chainer
import numpy as np
import os
from tqdm import tqdm
from brisque import BRISQUE
from mini_batch_loader import MiniBatchLoader
import State
from MyFCN import *
from pixelwise_a3c import *
import torch
import torch.optim as optim
from utils.traj_dataset import TrajectoryDataset
from utils.trajectory import list_of_tuple_to_traj
import cv2
import json
from utils.compute_Rbrisque import brisque_reward
from utils.augmentation import aug_mask

def overlapped_process(raw_x, mask, agent, current_state, patch_size, stride):
    _, _, h, w = raw_x.shape
    output = np.zeros_like(raw_x)
    counter = np.zeros_like(raw_x)
    for x in range(0, h, stride):
        for y in range(0, w, stride):
            x_end = min(x + patch_size, h)
            y_end = min(y + patch_size, w)
            patch_image = raw_x[:, :, x:x_end, y:y_end]
            patch_mask = mask[:, :, x:x_end, y:y_end]
            patch_output = inference_patch(agent, current_state, patch_image, patch_mask)
            output[:, :, x:x_end, y:y_end] += patch_output
            counter[:, :, x:x_end, y:y_end] += 1
    output = np.divide(output, counter)
    return output

def inference_patch(agent, current_state, patch_image, mask):
    # only return the final result
    current_state.reset(patch_image)
    mask_squeeze = np.squeeze(mask, axis=1)
    for t in range(0, args.episode_len):
        action, inner_state = agent.act(current_state.tensor)
        action = np.where(mask_squeeze==0, 1, action) # if mask equal to 0, the pixel isn't a rain, so the act should be id==1:"do nothing" 
        current_state.step(action, inner_state)  
    agent.stop_episode()
    return current_state.image

def inference(agent, raw_x, mask, name):
    os.makedirs(os.path.join(args.save_dir_path, 'derained_result'), exist_ok=True)
    current_state = State.State(args.move_range)
    B, C, H, W = raw_x.shape
    if H*W > 535000: # for high resolurion images, we use overlapped inference due to GPU limitations
        output = overlapped_process(raw_x, mask, agent, current_state, patch_size=128, stride=64)
        p = np.maximum(0,output)
        p = np.minimum(1,p)
        p = (p*255).astype(np.uint8)
        p = np.transpose(p[0], [1,2,0])
    else:
        current_state.reset(raw_x)
        mask_squeeze = np.squeeze(mask, axis=1)
        for t in range(0, args.episode_len):
            action, inner_state = agent.act(current_state.tensor)
            action = np.where(mask_squeeze==0, 1, action) # if mask equal to 0, the pixel isn't a rain, so the act should be id==1:"do nothing" 
            current_state.step(action, inner_state)
        agent.stop_episode()
            
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        p = (p*255).astype(np.uint8)
        p = np.transpose(p[0], [1,2,0])
    
    cv2.imwrite(os.path.join(args.save_dir_path, 'derained_result', name), p)

def compute_diff(image1, image2):
    return np.mean(np.abs(image1 - image2))

def main(args):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        args.data_path, 
        args.image_dir_path)
    
    brisque_metrics = BRISQUE(url=False)

    chainer.cuda.get_device_from_id(args.gpu_id).use()

    current_state = State.State(args.move_range)
    
    train_data_size = MiniBatchLoader.count_paths(args.data_path)
    
    # criterion for training Rnet
    CE = torch.nn.CrossEntropyLoss()
    
    for data_idx in range(0, train_data_size):
        # train
        raw_x, pseudo_ys, mask, name = mini_batch_loader.load_training_data(index=data_idx)
        model = MyFcn(args.n_actions)
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
        optimizer.setup(model)
        agent = PixelWiseA3C_InnerState(model, optimizer, 5, args.gamma)
        agent.act_deterministically = True
        agent.model.to_gpu()
        agent.load(os.path.join(args.checkpoint_dir_path, 'model_weight_best', name))
        print("===== Process for {} =====".format(name))
        # init Reward function
        r_net_brisque = Reward_Predictor(image_size=(args.pretrained_img_size, args.pretrained_img_size)).cuda()
        r_net_brisque.load_state_dict(torch.load(os.path.join(args.checkpoint_dir_path, 'model_weight_best', name, 'Rnet', 'rnet_brisque.pt')))
            
        inference(agent, raw_x, mask, name)

        agent.save(os.path.join(args.save_dir_path, 'model_weight', name))

        # save Rnet model
        os.makedirs(os.path.join(args.save_dir_path, 'model_weight', name, 'Rnet'), exist_ok=True)
        torch.save(r_net_brisque.state_dict(), os.path.join(args.save_dir_path, 'model_weight', name, 'Rnet', 'rnet_brisque.pt'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for training') 
    # seed
    parser.add_argument('--random_seed', type=int, default=1)
    # Directories
    parser.add_argument('--image_dir_path', type=str, default='dataset/')
    parser.add_argument('--data_path', type=str, default='dataset/Rain12/testing.txt')
    parser.add_argument('--gt_dir_path', type=str, default='dataset/Rain12/test/gt/')
    parser.add_argument('--save_dir_path', type=str, default='./Results/Rain12/test/SRL-Derain/')
    parser.add_argument('--checkpoint_dir_path', type=str, default='./Checkpoints/Rain12/SRL-Derain/')
    # config
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--move_range', type=int, default=3)
    parser.add_argument('--episode_len', type=int, default=15)
    parser.add_argument('--max_episode', type=int, default=150)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_actions', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ld', type=float, default=0.05, help='lambda, the weight for reward')
    parser.add_argument('--N_pre', type=int, default=6000)
    parser.add_argument('--pretrained_img_size', type=int, default=128)
    parser.add_argument('--pretrained_batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)