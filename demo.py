import os
import sys
import cv2
import random
import argparse

from model.model_builder import Detector

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def plot_one_box(img, x, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def demo_image(model, image_path, display):
    # 1. prepare image
    image = cv2.imread(image_path)

    # 2. model infer
    scores, boxes, classes = model.resize_infer(image)

    # 3. draw bounding box on the image
    for score, box, labels in zip(scores, boxes, classes):
        plot_one_box(
            image, box, COLORS[labels % 3]
        )  # , '{label}: {score:.3f}'.format(label=labels, score=score))

    # 4. visualize result
    if display:
        cv2.imshow("result", image)
        cv2.waitKey(0)
    else:
        path, _ = os.path.splitext(image_path)
        cv2.imwrite(path + "_result.jpg", image)
        print("output file save at '{}'".format(path + "_result.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "-m",
        "--model",
        help="the model name",
        choices=["fpn", "fpn+mp"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="optional checkpoint file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--demo-file",
        help="the address of the demo file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--display",
        help="whether display the detection result",
        action="store_true",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    model = Detector(args.model, args.checkpoint)
    demo_image(model, args.demo_file, args.display)