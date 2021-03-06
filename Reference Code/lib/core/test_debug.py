# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import pycocotools.mask as mask_util

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.image as image_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.keypoints as keypoint_utils
import pdb


def im_detect_all(model, im, box_proposals=None, timers=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)
    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        # raise NotImplementedError
        #scores, boxes, im_scale, blob_conv = im_detect_bbox_aug(model, im, box_proposals)
        scores, boxes, im_scale, _ = im_detect_bbox_aug(model, im, box_proposals)
    else:
        #scores, boxes, im_scale, blob_conv = im_detect_bbox(
        scores, boxes, im_scale, _ = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
    timers['im_detect_bbox'].toc()
    #blob_conv_net = model.module.Conv_Body
    # blob_conv = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    # scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    # pdb.set_trace()
    scores, boxes, cls_boxes, return_keep_np = box_results_with_nms_and_limit_return_keep(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask_amodal'].tic()
        timers['im_detect_mask_inmodal'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            #raise NotImplementedError
            masks = im_detect_mask_aug(model, im, boxes)
        else:
            blob_conv, _, amodal_feature, class_feature = im_conv_body_and_branch_feature(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
            # pdb.set_trace()
            class_feature_extract = class_feature[return_keep_np, :, :, :]
            amodal_feature_extract = amodal_feature[return_keep_np, :, :, :]

            if cfg.MODEL.INMODAL_ON:
                masks_amodal, masks_inmodal = im_detect_mask_and_branch_feature(model, im_scale, boxes, blob_conv, amodal_feature_extract, class_feature_extract)
            else:
                masks = im_detect_mask(model, im_scale, boxes, blob_conv)

        timers['im_detect_mask_amodal'].toc()
        timers['im_detect_mask_inmodal'].toc()

        timers['misc_mask_amodal'].tic()
        timers['misc_mask_inmodal'].tic()
        cls_segms_amodal, cls_segms_inmodal = segm_results_inmodal(cls_boxes, masks_amodal, masks_inmodal, boxes, im.shape[0], im.shape[1])
        timers['misc_mask_inmodal'].toc()
        timers['misc_mask_amodal'].toc()
    else:
        cls_segms_amodal = None
        cls_segms_inmodal = None

    return cls_boxes, cls_segms_amodal, cls_segms_inmodal

def im_detect_ensemble(model, im, cls_boxes, timers=None):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)

    #timers['misc_bbox'].tic()
    #scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    #timers['misc_bbox'].toc()
   
    num_classes = len(cls_boxes) 
    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
 
    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            #raise NotImplementedError
            masks = im_detect_mask_aug(model, im, boxes)
        else:
            blob_conv, im_scale = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms, ref_boxes = segm_results_ensemble(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None
        ref_boxes = None

    return cls_boxes, cls_segms, ref_boxes

def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']))]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']))]

    return_dict = model(**inputs)

    if cfg.MODEL.FASTER_RCNN:
        rois = return_dict['rois'].data.cpu().numpy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # cls prob (activations after softmax)
    scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = return_dict['bbox_pred'].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # (legacy) Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                         + cfg.TRAIN.BBOX_NORMALIZE_MEANS
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale, return_dict['blob_conv']


def im_detect_bbox_re(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']))]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']))]

    return_dict = model(**inputs)

    if cfg.MODEL.FASTER_RCNN:
        rois = return_dict['rois'].data.cpu().numpy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # cls prob (activations after softmax)
    scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = return_dict['bbox_pred'].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # (legacy) Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                         + cfg.TRAIN.BBOX_NORMALIZE_MEANS
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale, return_dict['blob_conv']



def im_detect_bbox_aug(model, im, box_proposals=None):
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model, 
            im, 
            cfg.TEST.SCALE, 
            cfg.TEST.MAX_SIZE, 
            box_proposals=box_proposals)
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals)
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True)
            add_preds_t(scores_scl_hf, boxes_scl_hf)


    # perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals, hflip=True)
        add_preds_t(scores_ar_hf, boxes_ar_hf)


    # Compute detections for original image(identity transform)
    scores_i, boxes_i, im_scale_i, blob_conv = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals)
    add_preds_t(scores_i, boxes_i)    
    
    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError('Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR))


    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
            )
    return scores_c, boxes_c, im_scale_i, blob_conv

def im_detect_bbox_hflip(model, im, target_scale, target_max_size, box_proposals=None):

    im_hf = im[:,::-1,:]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scale, _ = im_detect_bbox(model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf)

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)
    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
    model, im, target_scale, target_max_size, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _, = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl



def im_detect_bbox_aspect_ratio(
    model, im, aspect_ratio, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv

def im_detect_mask(model, im_scale, boxes, blob_conv):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    # pdb.set_trace()
    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    if not cfg.MODEL.INMODAL_ON:
        pred_masks = model.module.mask_net(blob_conv, inputs)
        pred_masks = pred_masks.data.cpu().numpy().squeeze()

        if cfg.MRCNN.CLS_SPECIFIC_MASK:
            pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
        else:
            pred_masks = pred_masks.reshape([-1, 1, M, M])

        return pred_masks
    else:
        pred_amodals, pred_inmodals = model.module.mask_net(blob_conv, inputs)
        pred_amodals = pred_amodals.data.cpu().numpy().squeeze()
        pred_inmodals = pred_inmodals.data.cpu().numpy().squeeze()

        if cfg.MRCNN.CLS_SPECIFIC_MASK:
            pred_amodals = pred_amodals.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
            pred_inmodals = pred_inmodals.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
        else:
            pred_amodals = pred_amodals.reshape([-1, 1, M, M])
            pred_inmodals = pred_inmodals.reshape([-1, 1, M, M])

        # pred_amodals = pred_amodals.reshape([-1, 1, M, M])

        return pred_amodals, pred_inmodals

def im_detect_mask_and_branch_feature(model, im_scale, boxes, blob_conv, amodal_feature, class_feature):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    # pdb.set_trace()
    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    if not cfg.MODEL.INMODAL_ON:
        pred_masks = model.module.mask_net(blob_conv, inputs, amodal_feature, class_feature)
        pred_masks = pred_masks.data.cpu().numpy().squeeze()

        if cfg.MRCNN.CLS_SPECIFIC_MASK:
            pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
        else:
            pred_masks = pred_masks.reshape([-1, 1, M, M])

        return pred_masks
    else:
        pred_amodals, pred_inmodals = model.module.mask_net(blob_conv, inputs, amodal_feature, class_feature)
        pred_amodals = pred_amodals.data.cpu().numpy().squeeze()
        pred_inmodals = pred_inmodals.data.cpu().numpy().squeeze()

        if cfg.MRCNN.CLS_SPECIFIC_MASK:
            pred_amodals = pred_amodals.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
            pred_inmodals = pred_inmodals.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
        else:
            pred_amodals = pred_amodals.reshape([-1, 1, M, M])
            pred_inmodals = pred_inmodals.reshape([-1, 1, M, M])

        # pred_amodals = pred_amodals.reshape([-1, 1, M, M])

        return pred_amodals, pred_inmodals

def im_detect_mask_aug(model, im, boxes):
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # compute masks for the original image (identity transform)
    blob_conv_i, im_scale_i = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    # _, im_scale_i, _ = blob_utils.get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    masks_i = im_detect_mask(model, im_scale_i, boxes, blob_conv_i)
    masks_ts.append(masks_i)

    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes)
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
       max_size = cfg.TEST.MASK_AUG.MAX_SIZE
       masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
       masks_ts.append(masks_scl)

       if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
           masks_scl_hf = im_detect_mask_scale(model, im, scale, max_size, boxes, hflip=True)
           masks_ts.append(masks_scl_hf)

    ## Compute masks at different aspect ratios
    #for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
    #   masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
    #   masks_ts.append(masks_ar)

    #   if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
    #       masks_ar_hf = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=True)
    #       masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks:
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0-y) / np.maximum(y, 1e-20))
        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError('Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR))
    return masks_c


def im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    # _, im_scale, _ = blob_utils.get_image_blob(im_hf, target_scale, target_max_size)
    blob_conv_hf, im_scale_hf = im_conv_body_only(model, im_hf, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    # im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    masks_hf = im_detect_mask(model, im_scale_hf, boxes_hf, blob_conv_hf)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes masks at the given scale"""
    if hflip:
        masks_scl = im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes)
    else:
        blob_conv_scl, im_scale_scl = im_conv_body_only(model, im, target_scale, target_max_size)
        # _, im_scale, _ = blob_utils.get_image_blob(im, target_scale, target_max_size)
        masks_scl = im_detect_mask(model, im_scale_scl, boxes, blob_conv_scl)
    return masks_scl

def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio"""

    # perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar)
    else:
        blob_conv_ar, im_scale_ar = im_conv_body_only(model, im, target_scale, target_max_size)
        # _, im_scale, _ = blob_utils.get_image_blob(im, target_scale, target_max_size)
        masks_ar = im_detect_mask(model, im_scale_ar, boxes_ar, blob_conv_ar)
    return masks_ar   

def im_detect_keypoints(model, im_scale, boxes, blob_conv):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    pred_heatmaps = model.module.keypoint_net(blob_conv, inputs)
    pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

def box_results_with_nms_and_limit_return_keep(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    return_keep = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        # pdb.set_trace()
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        keep = box_utils.nms(dets_j, cfg.TEST.NMS)
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets
        return_keep[j] = inds[keep]

    # Limit to max_per_image detections **over all classes**
    # if cfg.TEST.DETECTIONS_PER_IM > 0:
    #     image_scores = np.hstack(
    #         [cls_boxes[j][:, -1] for j in range(1, num_classes)]
    #     )
    #     if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
    #         image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
    #         for j in range(1, num_classes):
    #             keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
    #             return_keep[j] = return_keep[j][keep]
    #             cls_boxes[j] = cls_boxes[j][keep, :]

    # pdb.set_trace()
    return_keep_np_re = []
    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    for j in range(1, num_classes):
        return_keep_np_re.extend(list(return_keep[j]))

    return_keep_np = np.array(return_keep_np_re)
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes, return_keep_np


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle['counts'] = rle['counts'].decode('ascii')
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms

def segm_results_inmodal(cls_boxes, masks_amodal, masks_inmodal, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms_amodal_1 = [[] for _ in range(num_classes)]
    cls_segms_inmodal_1 = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_amodal = np.zeros((M + 2, M + 2), dtype=np.float32)
    padded_inmodal = np.zeros((M + 2, M + 2), dtype=np.float32)
    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms_amodal = []
        segms_inmodal = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_amodal[1:-1, 1:-1] = masks_amodal[mask_ind, j, :, :]
                padded_inmodal[1:-1, 1:-1] = masks_inmodal[mask_ind, j, :, :]
            else:
                padded_amodal[1:-1, 1:-1] = masks[mask_ind, 0, :, :]
                padded_inmodal[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask_amodal = cv2.resize(padded_amodal, (w, h))
            mask_amodal = np.array(mask_amodal > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask_amodal = np.zeros((im_h, im_w), dtype=np.uint8)

            mask_inmodal = cv2.resize(padded_inmodal, (w, h))
            mask_inmodal = np.array(mask_inmodal > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask_inmodal = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask_amodal[y_0:y_1, x_0:x_1] = mask_amodal[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            im_mask_inmodal[y_0:y_1, x_0:x_1] = mask_inmodal[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle_amodal = mask_util.encode(np.array(im_mask_amodal[:, :, np.newaxis], order='F'))[0]
            rle_inmodal = mask_util.encode(np.array(im_mask_inmodal[:, :, np.newaxis], order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle_amodal['counts'] = rle_amodal['counts'].decode('ascii')
            rle_inmodal['counts'] = rle_inmodal['counts'].decode('ascii')
            segms_amodal.append(rle_amodal)
            segms_inmodal.append(rle_inmodal)
            mask_ind += 1

        cls_segms_amodal_1[j] = segms_amodal
        cls_segms_inmodal_1[j] = segms_inmodal

    assert mask_ind == masks_amodal.shape[0]
    assert mask_ind == masks_inmodal.shape[0]
    return cls_segms_amodal_1, cls_segms_inmodal_1

def segm_results_ensemble(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            # mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            # im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            # x_0 = max(ref_box[0], 0)
            # x_1 = min(ref_box[2] + 1, im_w)
            # y_0 = max(ref_box[1], 0)
            # y_1 = min(ref_box[3] + 1, im_h)

            # im_mask[y_0:y_1, x_0:x_1] = mask[
                # (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            # rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            # rle['counts'] = rle['counts'].decode('ascii')
            segms.append(mask)

            mask_ind += 1

        cls_segms[j] = segms

    # assert mask_ind == masks.shape[0]
    return cls_segms, ref_boxes


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    if not cfg.FPN.PANET_ON:
        fpn_utils.add_multilevel_roi_blobs(blobs, name, blobs[name], lvls, lvl_min, lvl_max)
    else:
        fpn_utils.add_multilevel_roi_blobs_panet(blobs, name, blobs[name], lvl_min, lvl_max)


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale

def im_conv_body_only(model, im, target_scale, target_max_size):
    boxes = None
    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)
    conv_body_only = np.ones(1, dtype=np.int) 
    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
        inputs['conv_body_only'] = [Variable(torch.from_numpy(conv_body_only), volatile=True)]
    else:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']))]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']))]
        inputs['conv_body_only'] = [Variable(torch.from_numpy(conv_body_only))]
    blob_conv = model(**inputs)
    return blob_conv, im_scale

def im_conv_body_and_branch_feature(model, im, target_scale, target_max_size):
    boxes = None
    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)
    conv_body_only = np.ones(1, dtype=np.int) 
    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
        inputs['conv_body_only'] = [Variable(torch.from_numpy(conv_body_only), volatile=True)]
    else:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']))]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']))]
        inputs['conv_body_only'] = [Variable(torch.from_numpy(conv_body_only))]
    blob_conv, amodal_feature, class_feature = model(**inputs)
    return blob_conv, im_scale, amodal_feature, class_feature
