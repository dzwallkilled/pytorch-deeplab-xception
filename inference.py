import os
import numpy as np
import cv2
import tqdm
import argparse
import math

import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize, CenterCrop, Resize
from modeling.deeplab import DeepLab
from utils.utils import get_available_device
from utils.image_decomposition import decompose_image
from dataloaders.utils import get_rip_labels


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--rip_mode', type=str, default='patches-level2')
    parser.add_argument('--use_sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=800,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=800,
                        help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # cuda, seed and logging
    parser.add_argument('--gpus', type=int, default=1,
                        help='how many gpus to use (default=1)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    parser.add_argument('--exp_root', type=str, default='')
    args = parser.parse_args()

    args.device, args.cuda = get_available_device(args.gpus)

    nclass = 3

    model = DeepLab(num_classes=nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

    args.checkname = '/data2/data2/zewei/exp/RipData/DeepLabV3/patches/level2/CV5-1/model_best.pth.tar'
    ckpt = torch.load(args.checkname)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model = model.to(args.device)

    img_files = ['doc/tests/img_cv.png']
    out_file = 'doc/tests/img_seg.png'

    transforms = Compose([
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    color_map = get_rip_labels()
    img_cv = cv2.imread(img_files[0])
    pred = process_single_large_image(model, img_cv)
    mask = gen_mask(pred, nclass, color_map)
    out_img = composite_image(img_cv, mask, alpha=0.2)
    save_image(mask, out_file.split('.')[0] + f'_mask.png')
    save_image(out_img, out_file.split('.')[0] + f'_com.png')
    print(f'saved image {out_file}')

    with torch.no_grad():
        for img_file in img_files:
            name, ext = img_file.split('.')

            img_cv = cv2.imread(img_file)
            patches = decompose_image(img_cv, None, (800, 800), (300, 700))
            print(f'Decompose input image into {len(patches)} patches.')
            for i, patch in patches.items():
                img = transforms(patch.image)
                img = torch.stack([img], dim=0).cuda()

                output = model(img)
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)

                expanded_pred = torch.zeros()

                # out_img = output[0].cpu().permute((1, 2, 0)).numpy()
                # out_img = (out_img * 255).astype(np.uint8)
                mask = gen_mask(pred[0], nclass, color_map)
                out_img = composite_image(patch.image, mask, alpha=0.2)
                save_image(mask, name + f'_patch{i:02d}_seg.' + ext)
                save_image(out_img, name + f'_patch{i:02d}_seg_img.' + ext)
                print(f'saved image {out_file}')


def process_single_large_image(model, image, image_size=(1080, 1920), stride=(300, 700), patch_size=(800, 800)):
    model.eval()
    patches = decompose_image(image, None, patch_size, stride)
    transforms = Compose([
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    num_cols = math.ceil((image_size[1] - patch_size[1]) / stride[1]) + 1
    final_output = []
    with torch.no_grad():
        for patch_cnt, patch in patches.items():
            top_left_x = (patch_cnt % num_cols) * stride[1]
            top_left_y = (patch_cnt // num_cols) * stride[0]

            img = transforms(patch.image)
            img = torch.stack([img], dim=0).cuda()

            output = model(img)
            output = output.detach().cpu()
            expanded_output = torch.zeros((1, output.size(1)) + image_size, dtype=torch.float)
            y0 = max(0, top_left_y)
            y1 = min(top_left_y + patch_size[0], image_size[0])
            x0 = max(0, top_left_x)
            x1 = min(top_left_x + patch_size[1], image_size[1])
            expanded_output[:, :, y0:y1, x0:x1] = output[:, :, y0-top_left_y:y1-top_left_y, x0-top_left_x:x1-top_left_x]
            final_output.append(expanded_output)

    final_output = torch.cat(final_output)
    pred = final_output.max(0)[0].argmax(0)
    pred = pred.cpu().numpy()
    return pred


def composite_image(arr1, arr2, alpha=0.2):
    return arr1 * (1 - alpha) + arr2 * alpha


def gen_mask(array, nclass, color_map):
    r = array.copy()
    g = array.copy()
    b = array.copy()

    for i in range(nclass):
        r[array == i] = color_map[i][0]
        g[array == i] = color_map[i][1]
        b[array == i] = color_map[i][2]

    rgb = np.dstack((r, g, b))
    return rgb


def save_image(img, file_name):
    if img.dtype == 'uint8':
        cv2.imwrite(file_name, img)
    else:
        cv2.imwrite(file_name, img.astype('uint8'))


if __name__ == '__main__':
    main()