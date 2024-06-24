from abc import ABC, abstractmethod


__all__ = ["BaseTargetWithDenoising"]


class BaseTargetWithDenoising(ABC):
    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        super(BaseTargetWithDenoising, self).__init__()
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_metas = None

    @abstractmethod
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """
        Perform Hungarian matching between predictions and ground truth,
        returning the matched ground truth corresponding to the predictions
        along with the corresponding regression weights.
        """

    def get_dn_anchors(self, cls_target, box_target, *args, **kwargs):
        """
        Generate noisy instances for the current frame, with a total of
        'self.num_dn_groups' groups.
        """
        return None

    def update_dn(self, instance_feature, anchor, *args, **kwargs):
        """
        Insert the previously saved 'self.dn_metas' into the noisy instances
        of the current frame.
        """

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        """
        Randomly save information for 'self.num_temp_dn_groups' groups of
        temporal noisy instances to 'self.dn_metas'.
        """
        if self.num_temp_dn_groups < 0:
            return
        self.dn_metas = dict(dn_anchor=dn_anchor[:, : self.num_temp_dn_groups])
