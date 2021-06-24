import os
import sys
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as mask_util


def toCSV(csv_file, image_path, bboxes, classes, **kwargs):
    # open the csv file
    basepath, _ = os.path.split(csv_file)
    extra_keys = kwargs.keys()
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if not os.path.exists(csv_file):
        f = open(csv_file, "w+")
        f.write(
            "{image_path},{classes},{xmin},{ymin},{xmax},{ymax},{extra_keys}\n".format(
                image_path="filename",
                classes="class",
                extra_keys=",".join(extra_keys),
                xmin="xmin",
                ymin="ymin",
                xmax="xmax",
                ymax="ymax",
            )
        )
    else:
        f = open(csv_file, "a")

    extra_values = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]
    for i, bbx in enumerate(bboxes):
        extra_value = [str(extra_values[i][k]) for k in extra_keys]
        f.write(
            "{image_path},{classes},{xmin},{ymin},{xmax},{ymax},{extra_value}\n".format(
                image_path=image_path[i],
                classes=classes[i],
                extra_value=",".join(extra_value),
                xmin=bbx[0],
                ymin=bbx[1],
                xmax=bbx[2],
                ymax=bbx[3],
            )
        )
    f.close()


def convert_keypoints_json_to_detection_csv(input_json, output_csv):

    dataset = COCO(input_json)
    indexes = dataset.getImgIds()

    cats = dataset.loadCats(dataset.getCatIds())
    joints_names = cats[0]["keypoints"]
    # head_joints_index = [joints_names.index(name) for name in ['neck', 'head']]
    torse_joints_index = [
        joints_names.index(name)
        for name in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    ]

    image_path, heights, widths, bboxes, classes, trackId = [], [], [], [], [], []
    for index in tqdm(indexes):
        im_ann = dataset.loadImgs(index)
        annIds = dataset.getAnnIds(imgIds=index, iscrowd=None)
        objs = dataset.loadAnns(annIds)

        file_name = im_ann[0]["file_name"]
        height = im_ann[0]["height"]
        width = im_ann[0]["width"]
        for tid, obj in enumerate(objs):
            # remove no human
            if obj["category_id"] != 1:
                continue
            # remove no keypoints
            if "dp_masks" not in obj and obj["num_keypoints"] == 0:
                continue
            # remove no torse keypoints
            keypoints = np.array(obj["keypoints"]).reshape(-1, 3)
            joints_2d_vis = keypoints[:, 2]
            torse_joints_2d_vis = joints_2d_vis[torse_joints_index]
            if "dp_masks" not in obj and len(torse_joints_2d_vis.nonzero()[0]) == 0:
                continue

            bbr = obj["bbox"]
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            x2 = min([x2, width])
            y2 = min([y2, height])
            bbox = [x1, y1, x2, y2]

            bboxes.append(bbox), classes.append("person"), trackId.append(tid)
            image_path.append(file_name), heights.append(height), widths.append(width)

            if "dp_masks" in obj:
                # torso box
                if obj["dp_masks"][0]:
                    mask = mask_util.decode(obj["dp_masks"][0])
                    mask = cv2.resize(
                        mask,
                        (int(x2 - x1), int(y2 - y1)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    y_valid, x_valid = np.where(mask)
                    if len(x_valid) >= 2:
                        bbox = [
                            x_valid.min() + x1,
                            y_valid.min() + y1,
                            x_valid.max() + x1,
                            y_valid[-1].max() + y1,
                        ]
                        bboxes.append(bbox), classes.append("torso"), trackId.append(
                            tid
                        )
                        image_path.append(file_name), heights.append(
                            height
                        ), widths.append(width)

                # head box
                if obj["dp_masks"][13]:
                    mask = mask_util.decode(obj["dp_masks"][13])
                    mask = cv2.resize(
                        mask,
                        (int(x2 - x1), int(y2 - y1)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    y_valid, x_valid = np.where(mask)
                    if len(x_valid) >= 2:
                        bbox = [
                            x_valid.min() + x1,
                            y_valid.min() + y1,
                            x_valid.max() + x1,
                            y_valid[-1].max() + y1,
                        ]
                        bboxes.append(bbox), classes.append("head"), trackId.append(tid)
                        image_path.append(file_name), heights.append(
                            height
                        ), widths.append(width)

    if os.path.exists(output_csv):
        os.remove(output_csv)
    toCSV(output_csv, image_path, bboxes, classes, trackId=trackId)


def convert_keypoints_json_to_detection_easy_csv(input_json, output_csv):
    dataset = COCO(input_json)
    indexes = dataset.getImgIds()

    image_path, heights, widths, bboxes, classes, trackId = [], [], [], [], [], []
    for index in tqdm(indexes):
        im_ann = dataset.loadImgs(index)
        annIds = dataset.getAnnIds(imgIds=index, iscrowd=None)
        objs = dataset.loadAnns(annIds)

        file_name = im_ann[0]["file_name"]
        height = im_ann[0]["height"]
        width = im_ann[0]["width"]

        validate = [1 for obj in objs if "dp_masks" not in obj]
        if len(validate) != 0:
            continue

        for tid, obj in enumerate(objs):
            # person box
            bbr = obj["bbox"]
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            x2 = min([x2, width])
            y2 = min([y2, height])
            bbox = [x1, y1, x2, y2]

            bboxes.append(bbox), classes.append("person"), trackId.append(tid)
            image_path.append(file_name), heights.append(height), widths.append(width)

            # torso box
            if obj["dp_masks"][0]:
                mask = mask_util.decode(obj["dp_masks"][0])
                mask = cv2.resize(
                    mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST
                )
                y_valid, x_valid = np.where(mask)
                if len(x_valid) >= 2:
                    bbox = [
                        x_valid.min() + x1,
                        y_valid.min() + y1,
                        x_valid.max() + x1,
                        y_valid[-1].max() + y1,
                    ]
                    bboxes.append(bbox), classes.append("torso"), trackId.append(tid)
                    image_path.append(file_name), heights.append(height), widths.append(
                        width
                    )

            # head box
            if obj["dp_masks"][13]:
                mask = mask_util.decode(obj["dp_masks"][13])
                mask = cv2.resize(
                    mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST
                )
                y_valid, x_valid = np.where(mask)
                if len(x_valid) >= 2:
                    bbox = [
                        x_valid.min() + x1,
                        y_valid.min() + y1,
                        x_valid.max() + x1,
                        y_valid[-1].max() + y1,
                    ]
                    bboxes.append(bbox), classes.append("head"), trackId.append(tid)
                    image_path.append(file_name), heights.append(height), widths.append(
                        width
                    )

    if os.path.exists(output_csv):
        os.remove(output_csv)
    toCSV(output_csv, image_path, bboxes, classes, trackId=trackId)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("-o", "--output-dir", default=None, type=str, required=True)
    parser.add_argument(
        "-i",
        "--input-dir",
        help="the input dir for densepose dataset",
        default=None,
        type=str,
        required=True,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    for dset in ["train", "valminusminival", "minival"]:
        # hard dataset
        convert_keypoints_json_to_detection_csv(
            os.path.join(
                args.input_dir,
                f"DensePoseData/DensePose_COCO/densepose_coco_2014_{dset}.json",
            ),
            os.path.join(args.output_dir, f"coco_torso_hard_2014_{dset}.csv"),
        )
        # easy dataset
        convert_keypoints_json_to_detection_easy_csv(
            os.path.join(
                args.input_dir,
                f"DensePoseData/DensePose_COCO/densepose_coco_2014_{dset}.json",
            ),
            os.path.join(args.output_dir, f"coco_torso_easy_2014_{dset}.csv"),
        )
