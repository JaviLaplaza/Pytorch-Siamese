from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
# import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.siamese import SiameseNet

import torch


def main(im, bbox):
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    siam = SiameseNet(env.root_pretrained, design.net)
    
     
    
    with Image.fromarray(im) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]
    
    
    im = Image.fromarray(im)   
    
    torch.save(siam.state_dict(), '/home/nvidia/jlaplaza/siamfc_pytorch_test/siamfc_pretrained.pt')
        
    #gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
    pos_x, pos_y, target_w, target_h = region_to_bbox(bbox)
    
    print(target_w, target_h)
    # bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
    #                         filename, image, templates_z, scores, evaluation.start_frame)
    
        
        
        
    tracker(hp, run, design, im, pos_x, pos_y, target_w, target_h, final_score_sz,
            siam, evaluation.start_frame)
    
    
        
    """    
    _, precision, precision_auc, iou = _compile_results(gt, bboxes, evaluation.dist_threshold)
    
    
    
    print(evaluation.video + \
          ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision +\
          ' -- Precision AUC: ' + "%.2f" % precision_auc + \
          ' -- IOU: ' + "%.2f" % iou + \
          ' -- Speed: ' + "%.2f" % speed + ' --\n')
    """

def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)    

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    
    gt_name_list = [t for t in os.listdir(video_folder) if t.endswith(".txt")]
    gt_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in gt_name_list]
    gt_name_list.sort()
    gt = np.empty((1,4))
    gt = []
    for txt in gt_name_list:
        file=open(txt, "r")
        contents = file.read()
        _, _, x1, y1, x2, y2 = contents.split()
        gt.append([float(x1), float(y1), float(x2)-float(x1), float(y2)-float(y1)])
    
    #gt_file = os.path.join(video_folder, 'groundtruth.txt')
    #gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'
    
    return gt, frame_name_list, frame_sz, n_frames


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


if __name__ == '__main__':
    im = "test.jpg"
    #im = Image.open(im)

    bbox = [23, 266, 532, 704]
    sys.exit(main(im, bbox))
