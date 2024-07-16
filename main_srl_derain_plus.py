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
from utils.trajectory import list_of_tuple_to_traj
import cv2
from utils.compute_Rbrisque import brisque_reward

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

    for i in range(0, train_data_size):
        # train
        raw_x, pseudo_ys, mask, name = mini_batch_loader.load_training_data(index=i)
        model = MyFcn(args.n_actions)
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
        optimizer.setup(model)
        agent = PixelWiseA3C_InnerState(model, optimizer, 5, args.gamma)
        agent.act_deterministically = True
        agent.model.to_gpu()
        print("===== Process for {} =====".format(name))
        r_net_brisque = Reward_Predictor(image_size=(args.pretrained_img_size, args.pretrained_img_size)).cuda()
        rnet_brisque_state_dict = torch.load(os.path.join(args.rnet_weight_dir, 'rnet_brisque.pt'))
        r_net_brisque.load_state_dict(rnet_brisque_state_dict)

        print("----start training agent----")
        os.makedirs(os.path.join(args.save_dir_path, 'model_weight', name), exist_ok=True)
        for episode in tqdm(range(1, args.max_episode+1)):
            # random crop args.pretrained_img_size x args.pretrained_img_size
            img_size = args.pretrained_img_size
            _, c, h, w = raw_x.shape
            rand_range_h = h-img_size
            rand_range_w = w-img_size
            x_offset = np.random.randint(rand_range_w)
            y_offset = np.random.randint(rand_range_h)
            raw_x_crop = raw_x[:, :, y_offset:y_offset+img_size, x_offset:x_offset+img_size]
            current_state.reset(raw_x_crop)
            mask_crop = mask[:, :, y_offset:y_offset+img_size, x_offset:x_offset+img_size]
            # random sample pseudo_y from 50 pseudo_ys
            rand_index = np.random.randint(0, 50)
            pseudo_y = np.expand_dims(pseudo_ys[rand_index], axis=0)
            # random crop img_size x img_size
            pseudo_y_crop = pseudo_y[:, :, y_offset:y_offset+img_size, x_offset:x_offset+img_size]
            reward = np.zeros(pseudo_y_crop.shape, pseudo_y_crop.dtype)
            # current_eps stores the transitions which belong to the same episode.
            current_eps = []
            sum_reward = 0
            sum_reward_brisque_rnet = 0
            for t in range(0, args.episode_len):
                previous_image = current_state.image.copy()
                action, inner_state = agent.act_and_train(current_state.tensor, reward)
                current_state.step_with_mask(action, mask_crop, inner_state)
                reward = np.square(pseudo_y_crop - previous_image)*255 - np.square(pseudo_y_crop - current_state.image)*255
                # add transition_tuple to current_eps
                # transition_tuple: (state_t[3, h, w], reward_brisque[1], state_t+1[3, h, w])
                reward_brisque = brisque_reward(brisque_metrics, previous_image.copy(), current_state.image.copy())
                transition_tuple = (previous_image.copy().squeeze(), reward_brisque*np.power(args.gamma,t), current_state.image.copy().squeeze())
                current_eps.append(transition_tuple)
                # generate trajectory and compute Rnet reward
                traj = list_of_tuple_to_traj(current_eps)
                with torch.no_grad():
                    input_st = traj.states
                    input_st = torch.tensor(input_st).cuda()
                    brisque_reward_rnet = r_net_brisque(input_st)[0].detach().cpu().numpy()
                    reward += args.ld * brisque_reward_rnet
                    sum_reward_brisque_rnet += brisque_reward_rnet*np.power(args.gamma,t)
                sum_reward += np.mean(reward)*np.power(args.gamma,t)        
            agent.stop_episode_and_train(current_state.tensor, reward, True)
            optimizer.alpha = args.lr*((1-episode/args.max_episode)**0.9)
            agent.save(os.path.join(args.save_dir_path, 'model_weight', name))
            
        inference(agent, raw_x, mask, name)
    
        # save Rnet model
        os.makedirs(os.path.join(args.save_dir_path, 'model_weight', name, 'Rnet'), exist_ok=True)
        torch.save(r_net_brisque.state_dict(), os.path.join(args.save_dir_path, 'model_weight', name, 'Rnet', 'rnet_brisque.pt'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for training') 
    # seed
    parser.add_argument('--random_seed', type=int, default=1)
    # Directories
    parser.add_argument('--image_dir_path', type=str, default='dataset/')
    parser.add_argument('--data_path', type=str, default='dataset/Rain100L/testing.txt')
    parser.add_argument('--save_dir_path', type=str, default='./Results/Rain100L/test/SRL-Derain+/')
    parser.add_argument('--rnet_weight_dir', type=str, default='./Results/Rain100L/test/Rnet+/model_weight/')
    # config
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--move_range', type=int, default=3)
    parser.add_argument('--episode_len', type=int, default=15)
    parser.add_argument('--max_episode', type=int, default=150)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_actions', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ld', type=float, default=0.05, help='lambda, the weight for reward')
    parser.add_argument('--pretrained_img_size', type=int, default=128)
    args = parser.parse_args()

    main(args)