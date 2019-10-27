import os
import cv2
import numpy as np
import pickle
import torch


def get_cam(model, target=None, image=None, args=None):
    """
    Return CAM tensor which shape is (batch, 1, h, w)
    """
    with torch.no_grad():

        if image is not None:
            _ = model(image)

        # Extract feature map
        if args.distributed:
            feature_map, score = model.module.get_cam()
        else:
            feature_map, score = model.get_cam()

        # Extract fc weight
        batch, channel, _, _ = feature_map.size()

        # get prediction in shape (batch)
        if target is None:
            _, target = score.topk(1, 1, True, True)
        target = target.squeeze()

        cam = feature_map[range(batch), target]
        return cam.unsqueeze(1)


def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def blend_cam(image, cam):
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.5 + heatmap * 0.5

    return blend, heatmap


def load_bbox(args):
    """ Load bounding box information """
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}

    dataset_path = args.data_list
    resize_size = args.resize_size
    crop_size = args.crop_size

    if args.dataset == 'CUB':
        with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
            for each_line in f:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])

                x, y, bbox_width, bbox_height = map(float, file_info[1:])

                origin_bbox[image_id] = [x, y, bbox_width, bbox_height]

        with open(os.path.join(dataset_path, 'sizes.txt')) as f:
            for each_line in f:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])
                image_width, image_height = map(float, file_info[1:])

                image_sizes[image_id] = [image_width, image_height]

        if args.VAL_CROP:
            shift_size = (resize_size - crop_size) // 2
        else:
            resize_size = crop_size
            shift_size = 0
        for key in origin_bbox.keys():
            x, y, bbox_width, bbox_height = origin_bbox[key]
            image_width, image_height = image_sizes[key]
            left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))

            right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, crop_size - 1))
            right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, crop_size - 1))
            resized_bbox[key] = [[left_bottom_x, left_bottom_y, right_top_x, right_top_y]]

    elif args.dataset == 'ILSVRC':
        with open(os.path.join(dataset_path, 'gt_ImageNet.pickle'), 'rb') as f:
            info_imagenet = pickle.load(f)

        origin_bbox = info_imagenet['gt_bboxes']
        image_sizes = info_imagenet['image_sizes']
        if args.VAL_CROP:
            shift_size = (resize_size - crop_size) // 2
        else:
            resize_size = crop_size
            shift_size = 0
        for key in origin_bbox.keys():
            image_height, image_width = image_sizes[key]
            resized_bbox[key] = list()
            for bbox in origin_bbox[key]:
                x_min, y_min, x_max, y_max = bbox
                left_bottom_x = int(max(x_min / image_width * resize_size - shift_size, 0))
                left_bottom_y = int(max(y_min / image_height * resize_size - shift_size, 0))
                right_top_x = int(min(x_max / image_width * resize_size - shift_size, crop_size-1))
                right_top_y = int(min(y_max / image_height * resize_size - shift_size, crop_size-1))

                resized_bbox[key].append([left_bottom_x, left_bottom_y, right_top_x, right_top_y])
    else:
        raise Exception("No dataset named {}".format(args.dataset))

    return resized_bbox


def get_bboxes(cam, cam_thr=0.2):
    """
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1)
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)

    return estimated bounding box, blend image with boxes
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_BINARY)

    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox




