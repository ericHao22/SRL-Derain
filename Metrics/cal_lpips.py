import lpips
from PIL import Image
import os
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
transform2=transforms.Compose([transforms.ToTensor()])

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

if __name__ == "__main__":
    gt_path = './dataset/Rain800/test/gt/'
    im_path ='./Results/Rain800/test/SRL-Derain/derained_result/'
    #im_path = './out(convolution,^6,ks=3)/'
    gt_folder = os.listdir(gt_path)
    im_folder = os.listdir(im_path)
    
    sum =0
    num =0
    
    for i in range(len(gt_folder)):
        gt=Image.open(gt_path+gt_folder[i]).convert('RGB')
        tensor_gt=transform2(gt)
        im=Image.open(im_path+gt_folder[i]).convert('RGB')
        tensor_im=transform2(im)

        sum += loss_fn_alex(tensor_gt, tensor_im).item()
        num += 1

    print(sum/num)