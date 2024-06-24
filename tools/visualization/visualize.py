import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender

plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = True, # True: draw history tracked boxes
    motion = True,
    map = True,
    planning = True,
)
START = 0
END = 81
INTERVAL = 1


class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        
        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)
        self.results = mmcv.load(args.result_path)
        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]['img_bbox']

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)
    
    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='vis',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(START, END, INTERVAL)):
        if idx > len(visualizer.results):
            break
        visualizer.add_vis(idx)
    
    visualizer.image2video()

if __name__ == '__main__':
    main()