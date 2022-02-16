import json
import os
import tqdm
import numpy as np
from retinaface.src.retinaface import RetinaFace

def load_data(pred_dir):
    events = os.listdir(pred_dir)
    pbar = tqdm.tqdm(events)
    directory = {}

    for event in pbar:
        pbar.set_description('Reading Predictions {}'.format(event))
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        path_list = []
        for imgname in event_images:
            path_list.append(f"{event_dir}/{imgname}")
        directory[event] = path_list
    return directory


def inference(path_dict):
    fd = RetinaFace(quality='high')
    result = {}
    if type(path_dict) == dict:
        for group in path_dict.keys():
            current_event = {}
            pbar = tqdm.tqdm(path_dict[group])

            for path in pbar:
                pbar.set_description(f"FaceDetection Processing...")
                img = fd.read(path)
                pred = fd.predict(img, threshold=0.8)
                current_event[path.split('/')[-1].rstrip('.jpg')] = np.array([[r['x1'], r['y1'],
                                                                                  r['x2']-r['x1'],
                                                                                  r['y2']-r['y1'],
                                                                                  0.9] for r in pred])
            result[group] = current_event
        return result
    else:
        current_event = {}
        img = fd.read(path_dict)
        pred = fd.predict(img, threshold=0.8)
        current_event[path_dict.split('/')[-1].rstrip('.jpg')] = np.array([[r['x1'], r['y1'],
                                                                       r['x2'] - r['x1'],
                                                                       r['y2'] - r['y1'],
                                                                       0.9] for r in pred])

        return current_event



def calculate_map_score(pred, gt_path, iou_thresh=0.5):
    def image_eval(pred, gt, ignore, iou_thresh):
        def bbox_overlaps(
                pred,  # pred
                gt):

            N = pred.shape[0]
            K = gt.shape[0]
            boxes = pred.copy()
            query_boxes = gt.copy()
            overlaps = np.zeros((N, K))

            for k in range(K):
                box_area = (
                        (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                        (query_boxes[k, 3] - query_boxes[k, 1] + 1)
                )
                for n in range(N):
                    iw = (
                            min(boxes[n, 2], query_boxes[k, 2]) -
                            max(boxes[n, 0], query_boxes[k, 0]) + 1
                    )
                    if iw > 0:
                        ih = (
                                min(boxes[n, 3], query_boxes[k, 3]) -
                                max(boxes[n, 1], query_boxes[k, 1]) + 1
                        )
                        if ih > 0:
                            ua = float(
                                (boxes[n, 2] - boxes[n, 0] + 1) *
                                (boxes[n, 3] - boxes[n, 1] + 1) +
                                box_area - iw * ih
                            )
                            overlaps[n, k] = iw * ih / ua
            return overlaps

        _pred = pred.copy()
        _gt = gt.copy()
        pred_recall = np.zeros(_pred.shape[0])
        recall_list = np.zeros(_gt.shape[0])
        proposal_list = np.ones(_pred.shape[0])

        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]


        overlaps = bbox_overlaps(_pred[:, :4], _gt)

        for h in range(_pred.shape[0]):

            gt_overlap = overlaps[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
            if max_overlap >= iou_thresh:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1

            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        return pred_recall, proposal_list

    def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
        pr_info = np.zeros((thresh_num, 2)).astype('float')
        for t in range(thresh_num):

            thresh = 1 - (t + 1) / thresh_num
            r_index = np.where(pred_info[:, 4] >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        return pr_info

    def dataset_pr_info(thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            if (pr_curve[i, 1]==0) and (pr_curve[i, 0]==0):
                _pr_curve[i, 0] = 0
                _pr_curve[i, 1] = pr_curve[i, 1] / count_face
            else:
                _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
                _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve

    def voc_ap(rec, prec):

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    with open(gt_path, 'r') as f:
        gt_dict = json.load(fp=f)

    mean_ap = []

    for group in pred.keys():
        check_group = group
        pred_list = pred[check_group]
        gt_list = gt_dict[check_group]
        thresh_num = 1000
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')

        for name in pred_list.keys():
            pred_info = pred_list[name]
            gt_boxes = np.array(gt_list[name]['gt_bbx_list']).astype('float')
            keep_index = np.array(gt_list[name]['gt_index'])
            count_face += len(keep_index)

            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue
            ignore = np.zeros(gt_boxes.shape[0])
            if len(keep_index) != 0:
                ignore[keep_index - 1] = 1
            pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
            _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
            pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        #
        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]
        #
        mean_ap.append(voc_ap(recall, propose))

    return np.mean(mean_ap)


if __name__ == '__main__':

    path_dict = load_data('wider_val')
    res = inference(path_dict)
    print(calculate_map_score(res, 'src/data/ground_truth/wider_medium_val.json'))