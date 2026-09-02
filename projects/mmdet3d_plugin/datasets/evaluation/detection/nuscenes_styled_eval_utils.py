from collections import defaultdict
from typing import List, Dict, Tuple, Union, Callable
import abc
import numpy as np
from pyquaternion import Quaternion


def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))


def velocity_l2(gt_box, pred_box) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity))


def yaw_diff(gt_box, eval_box, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def attr_acc(gt_box, pred_box) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc


def scale_iou(sample_annotation, sample_result) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw



def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)
    

class DetectionMetricData(abc.ABC):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 precision: np.array,
                 confidence: np.array,
                 trans_err: np.array,
                 vel_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 attr_err: np.array):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']))

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   vel_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   attr_err=np.ones(cls.nelem))

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1],
                   trans_err=np.random.random(cls.nelem),
                   vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem),
                   orient_err=np.random.random(cls.nelem),
                   attr_err=np.random.random(cls.nelem))
    

class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[DetectionMetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[DetectionMetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMetricData.deserialize(md))
        return mdl

class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: dict):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in self.cfg['tp_metrics']:
            class_errors = []
            for detection_name in self.cfg['class_names']:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in self.cfg['tp_metrics']:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        # Summarize.
        total = float(self.cfg['mean_ap_weight'] * self.mean_ap + np.sum(list(self.tp_scores.values())))

        # Normalize.
        total = total / float(self.cfg['mean_ap_weight'] + len(self.tp_scores.keys()))

        return total
    

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = content['cfg']
        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq
    

class DetectionBox(abc.ABC):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.


        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        # assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        # assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
        #     'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.num_pts = num_pts
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])
    @property
    def ego_dist(self) -> float:
        """ Compute the distance from this box to the ego vehicle in 2D. """
        return np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2))    





class EvalBoxes:
    """ Data class that groups EvalBox instances by sample. """

    def __init__(self):
        """
        Initializes the EvalBoxes for GT or predictions.
        """
        self.boxes = defaultdict(list)

    def __repr__(self):
        return "EvalBoxes with {} boxes across {} samples".format(len(self.all), len(self.sample_tokens))

    def __getitem__(self, item) -> List[DetectionBox]:
        return self.boxes[item]

    def __eq__(self, other):
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    def __len__(self):
        return len(self.boxes)

    @property
    def all(self) -> List[DetectionBox]:
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        """ Returns a list of all keys. """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[DetectionBox]) -> None:
        """ Adds a list of boxes. """
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        """
        Initialize from serialized content.
        :param content: A dictionary with the serialized content of the box.
        :param box_cls: The class of the boxes, DetectionBox or TrackingBox.
        """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
        return eb
    

def accumulate(gt_boxes,
               pred_boxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])



def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
    
    
def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw