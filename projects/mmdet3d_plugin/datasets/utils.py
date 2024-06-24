import copy

import cv2
import numpy as np
import torch

from projects.mmdet3d_plugin.core.box3d import *


def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners


def plot_rect3d_on_img(
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if isinstance(color[0], int):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    # corners_3d = bboxes3d.corners
    corners_3d = box3d_to_corners(bboxes3d)
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3):
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.zeros([bev_h, bev_w, 3])

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),
            int((cir + 1) * 10 / bev_resolution),
            marking_color,
            thickness=thickness,
        )
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
    )
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
    )
    if len(bboxes_3d) != 0:
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][
            ..., [0, 1]
        ]
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                )
    return bev.astype(np.uint8)


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
        vis_imgs.append(
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )

    num_imgs = len(vis_imgs)
    if num_imgs < 4 or num_imgs % 2 != 0:
        vis_imgs = np.concatenate(vis_imgs, axis=1)
    else:
        vis_imgs = np.concatenate([
            np.concatenate(vis_imgs[:num_imgs//2], axis=1),
            np.concatenate(vis_imgs[num_imgs//2:], axis=1)
        ], axis=0)

    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
    vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
    return vis_imgs
