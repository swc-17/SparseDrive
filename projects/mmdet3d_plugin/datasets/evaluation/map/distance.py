from scipy.spatial import distance
from numpy.typing import NDArray
import torch

def chamfer_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate chamfer distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): chamfer distance
    '''
    
    dist_matrix = distance.cdist(line1, line2, 'euclidean')
    dist12 = dist_matrix.min(-1).sum() / len(line1)
    dist21 = dist_matrix.min(-2).sum() / len(line2)

    return (dist12 + dist21) / 2

def frechet_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate frechet distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): frechet distance
    '''
    
    raise NotImplementedError

def chamfer_distance_batch(pred_lines, gt_lines):
    ''' Calculate chamfer distance between two group of lines. Make sure the 
    lines are interpolated.

    Args:
        pred_lines (array or tensor): shape (m, num_pts, 2 or 3)
        gt_lines (array or tensor): shape (n, num_pts, 2 or 3)
    
    Returns:
        distance (array): chamfer distance
    '''
    _, num_pts, coord_dims = pred_lines.shape
    
    if not isinstance(pred_lines, torch.Tensor):
        pred_lines = torch.tensor(pred_lines)
    if not isinstance(gt_lines, torch.Tensor):
        gt_lines = torch.tensor(gt_lines)
    dist_mat = torch.cdist(pred_lines.view(-1, coord_dims), 
                    gt_lines.view(-1, coord_dims), p=2) 
    # (num_query*num_points, num_gt*num_points)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts)) 
    # (num_query, num_points, num_gt*num_points)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts, dim=-1)) 
    # (num_gt, num_q, num_pts, num_pts)

    dist1 = dist_mat.min(-1)[0].sum(-1)
    dist2 = dist_mat.min(-2)[0].sum(-1)

    dist_matrix = (dist1 + dist2).transpose(0, 1) / (2 * num_pts)
    
    return dist_matrix.numpy()