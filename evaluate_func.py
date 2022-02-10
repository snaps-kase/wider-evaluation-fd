import requests
import base64, json
import argparse
import os
import tqdm
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps


def load_data(pred_dir):
    events = os.listdir(pred_dir)
    pbar = tqdm.tqdm(events)

    path_list = []
    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        for imgname in event_images:
            path_list.append(f"{event_dir}/{imgname}")
    return path_list



def request_fd_api(path_list):
    pred = dict()
    current_event = dict()

    url = ""

    if type(path_list) == str:
        dir, file = os.path.split(path_list)
        dir = dir.split('/')[-1]
        path_list = [path_list]
    else:
        dir, file = os.path.split(path_list[0])
        dir = dir.split('/')[-1]

    pbar_path = tqdm.tqdm(path_list)

    for imgname in pbar_path:
        pbar_list.set_description(f"FD_Processing...")
        jpgtxt = base64.b64encode(open(f"{imgname}", "rb").read()).decode('utf-8')
        data = {
            "imageData": jpgtxt,
            "types": "bytes",
            "ot": 1
        }
        response = requests.post(url=url, json=data)
        result = json.loads(response.text)
        current_event[imgname.split('/')[-1].rstrip('.jpg')] = np.array([[r['x1'], r['y1'],
                                                                          r['x2']-r['x1'],
                                                                          r['y2']-r['y1'],
                                                                        r['confidence']] for r in result['faces']])

    pred[dir] = current_event
    return pred



def calculate_score(pred, gt_path, iou_thresh=0.5):
    def get_gt_boxes(gt_dir):
        """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

        gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
        hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
        medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
        easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

        facebox_list = gt_mat['face_bbx_list']
        event_list = gt_mat['event_list']
        file_list = gt_mat['file_list']

        hard_gt_list = hard_mat['gt_list']
        medium_gt_list = medium_mat['gt_list']
        easy_gt_list = easy_mat['gt_list']

        return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

    def image_eval(pred, gt, ignore, iou_thresh):
        """ single image evaluation
        pred: Nx5
        gt: Nx4
        ignore:
        """

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

    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['medium']
    setting_gts = medium_gt_list
    gt_list = setting_gts
    count_face = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    # [hard, medium, easy]
    pbar = tqdm.tqdm(range(event_num))

    for i in pbar:
        pbar.set_description('Processing {}'.format(settings[0]))
        event_name = str(event_list[i][0][0])
        img_list = file_list[i][0]
        if event_name in pred.keys():
            pred_list = pred[event_name]
        else:
            continue
        sub_gt_list = gt_list[i][0]
        gt_bbx_list = facebox_list[i][0]

        for j in range(len(img_list)):
            if str(img_list[j][0][0]) not in pred_list.keys():
                continue
            pred_info = pred_list[str(img_list[j][0][0])]
            gt_boxes = gt_bbx_list[j][0].astype('float')
            keep_index = sub_gt_list[j][0]
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

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        mean_ap = voc_ap(recall, propose)

    print("==================== Results ====================")
    print("Medium  Val mAP: {}".format(mean_ap))
    print("=================================================")
    return mean_ap




if __name__ == '__main__':
    path_list = load_data('./wider_val')

    print(path_list[133:141][0])
    result = request_fd_api(path_list)
    print(calculate_score(result, gt_path='./src/data/ground_truth', iou_thresh=0.5))