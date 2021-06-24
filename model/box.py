import torch


def configure_ratio_scale(num_featmaps, ratios, scales):
    # for current version do not consider the enlarged 1:1 anchor box in the original ssd or tf version
    if len(scales) == num_featmaps:
        scales = scales
    # for the current version of generate anchor, this is not make sense
    # elif len(cfg.SIZES) == 2:
    #     num_layers = len(strides)
    #     min_scale, max_scale = cfg.SIZES
    #     scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)]
    else:
        raise ValueError(
            "cfg.SIZES is not correct,"
            "the len of cfg.SIZES should equal to num layers({}) or 2, but it is {}".format(
                num_featmaps, len(scales)
            )
        )
    for i in range(num_featmaps):
        if not isinstance(scales[i], list):
            scales[i] = [scales[i]]

    if isinstance(ratios[0], list):
        if len(ratios) == num_featmaps:
            ratios = ratios
        else:
            raise ValueError(
                "When cfg.ASPECT_RATIOS contains list for each layer,"
                "Len of cfg.ASPECT_RATIOS should equal to num layers({}), but it is {}".format(
                    num_featmaps, len(ratios)
                )
            )
    else:
        ratios = [ratios for _ in range(num_featmaps)]
    return ratios, scales


def generate_anchors(stride, ratio_vals, scales_vals):
    "Generate anchors coordinates from scales/ratios"

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.round(torch.sqrt(wh[:, 0] * wh[:, 1] / ratios))
    dwh = torch.stack([ws, torch.round(ws * ratios)], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales) - 1
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):
    "Convert boxes to deltas from anchors"

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[..., 2:] - boxes[..., :2] + 1
    boxes_ctr = boxes[..., :2] + 0.5 * boxes_wh

    return torch.cat(
        [(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], -1
    )


def delta2box(deltas, anchors, size, stride):
    "Convert deltas from anchors to boxes"

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat(
        [clamp(pred_ctr - 0.5 * pred_wh), clamp(pred_ctr + 0.5 * pred_wh - 1)], 1
    )


def decode(
    all_cls_head,
    all_box_head,
    stride=1,
    threshold=0.05,
    top_n=1000,
    anchors=None,
    rescore=True,
):
    "Box Decoding and Filtering"

    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = -1 * torch.ones((batch_size, top_n), device=device)
    out_boxes = -1 * torch.ones((batch_size, top_n, 4), device=device)
    out_classes = -1 * torch.ones((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, 4)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero(as_tuple=False).view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices // width // height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices // width) % height
        a = indices // num_classes // height // width
        box_head = box_head.view(num_anchors, 4, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = (
                torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride
                + anchors[a, :]
            )
            boxes = delta2box(boxes, grid, [width, height], stride)
            if rescore:
                grid_center = (grid[:, :2] + grid[:, 2:]) / 2
                lt = torch.abs(grid_center - boxes[:, :2])
                rb = torch.abs(boxes[:, 2:] - grid_center)
                centerness = torch.sqrt(
                    torch.prod(torch.min(lt, rb) / torch.max(lt, rb), dim=1)
                )
                scores = scores * centerness

        out_scores[batch, : scores.size()[0]] = scores
        out_boxes[batch, : boxes.size()[0], :] = boxes
        out_classes[batch, : classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100, using_diou=True):
    "Non Maximum Suppression"

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = -1 * torch.ones((batch_size, ndetections), device=device)
    out_boxes = -1 * torch.ones((batch_size, ndetections, 4), device=device)
    out_classes = -1 * torch.ones((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero(as_tuple=False)
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(
            -1
        )
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero(as_tuple=False).nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)
            iou = inter / (areas + areas[i] - inter + 1e-7)

            if using_diou:
                outer_lt = torch.min(boxes[:, :2], boxes[i, :2])
                outer_rb = torch.max(boxes[:, 2:], boxes[i, 2:])

                inter_diag = ((boxes[:, :2] - boxes[i, :2]) ** 2).sum(dim=1)
                outer_diag = ((outer_rb - outer_lt) ** 2).sum(dim=1) + 1e-7
                diou = (iou - inter_diag / outer_diag).clamp(-1.0, 1.0)
                iou = diou

            criterion = (scores > scores[i]) | (iou <= nms) | (classes != classes[i])
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero(as_tuple=False)].view(-1)
            boxes = boxes[criterion.nonzero(as_tuple=False), :].view(-1, 4)
            classes = classes[criterion.nonzero(as_tuple=False)].view(-1)
            areas = areas[criterion.nonzero(as_tuple=False)].view(-1)
            keep[(~criterion).nonzero(as_tuple=False)] = 0

        out_scores[batch, : i + 1] = scores[: i + 1]
        out_boxes[batch, : i + 1, :] = boxes[: i + 1, :]
        out_classes[batch, : i + 1] = classes[: i + 1]

    return out_scores, out_boxes, out_classes


def set_decode(
    all_cls_head,
    all_box_head,
    all_box_shift_head,
    stride=1,
    threshold=0.05,
    top_n=1000,
    anchors=None,
):
    "Box Decoding and Filtering"

    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    num_boxes = all_box_shift_head.size()[1] // num_anchors // 4 + 1
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = -1 * torch.ones((batch_size, top_n), device=device)
    out_boxes = -1 * torch.ones((batch_size, top_n, num_boxes, 4), device=device)
    out_classes = -1 * torch.ones((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, 4)
        sbox_head = all_box_shift_head[batch, :, :, :].contiguous().view(-1, 4)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero(as_tuple=False).view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices // width // height) % num_classes
        classes = classes.type(all_cls_head.type())
        # scores  = torch.stack((scores, scores)).permute(1,0).reshape(-1)
        # classes = torch.stack((classes*2, classes*2+1)).permute(1,0) #.reshape(-1)

        # Infer kept bboxes
        x = indices % width
        y = (indices // width) % height
        a = indices // num_classes // height // width
        box_head = box_head.view(num_anchors, 4, height, width)
        boxes = box_head[a, :, y, x]
        sbox_head = sbox_head.view(num_anchors, 4, height, width)
        sboxes = sbox_head[a, :, y, x]

        if anchors is not None:
            grid = (
                torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride
                + anchors[a, :]
            )
            boxes = delta2box(boxes, grid, [width, height], stride)
            sboxes = delta2box(sboxes, grid, [width, height], stride)
        boxes = torch.stack((boxes, sboxes)).permute(1, 0, 2)  # .reshape(-1, 4)

        out_scores[batch, : scores.size()[0]] = scores
        out_boxes[batch, : boxes.size()[0], :] = boxes
        out_classes[batch, : classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def set_nms(
    all_scores, all_boxes, all_classes, nms=0.5, ndetections=100, using_diou=True
):
    "Non Maximum Suppression"

    device = all_scores.device
    batch_size = all_scores.size()[0]
    num_boxes = all_boxes.size()[2]

    out_scores = -1 * torch.ones((batch_size, ndetections * num_boxes), device=device)
    out_boxes = -1 * torch.ones((batch_size, ndetections * num_boxes, 4), device=device)
    out_classes = -1 * torch.ones((batch_size, ndetections * num_boxes), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero(as_tuple=False)
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, num_boxes, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, :, 2] - boxes[:, :, 0] + 1) * (
            boxes[:, :, 3] - boxes[:, :, 1] + 1
        ).view(-1, num_boxes)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero(as_tuple=False).nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :, :2], boxes[i, :, :2])
            xy2 = torch.min(boxes[:, :, 2:], boxes[i, :, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 2)
            iou = inter / (areas + areas[i] - inter + 1e-7)

            if using_diou:
                outer_lt = torch.min(boxes[:, :, :2], boxes[i, :, :2])
                outer_rb = torch.max(boxes[:, :, 2:], boxes[i, :, 2:])

                inter_diag = ((boxes[:, :, :2] - boxes[i, :, :2]) ** 2).sum(dim=2)
                outer_diag = ((outer_rb - outer_lt) ** 2).sum(dim=2) + 1e-7
                diou = (iou - inter_diag / outer_diag).clamp(-1.0, 1.0)
                iou = diou

            # find the min iou in the bounding box group
            iou, _ = torch.max(iou, dim=1)
            # iou, _ = torch.min(iou, dim=1)
            criterion = (scores > scores[i]) | (iou <= nms) | (classes != classes[i])
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero(as_tuple=False)].view(-1)
            boxes = boxes[criterion.nonzero(as_tuple=False), :].view(-1, num_boxes, 4)
            classes = classes[criterion.nonzero(as_tuple=False)].view(-1)
            areas = areas[criterion.nonzero(as_tuple=False)].view(-1, num_boxes)
            keep[(~criterion).nonzero(as_tuple=False)] = 0

        scores = torch.stack((scores, scores)).permute(1, 0).reshape(-1)
        classes = torch.stack((classes * 2, classes * 2 + 1)).permute(1, 0).reshape(-1)
        boxes = boxes.reshape(-1, 4)
        i = i * 2 + 1

        out_scores[batch, : i + 1] = scores[: i + 1]
        out_boxes[batch, : i + 1, :] = boxes[: i + 1, :]
        out_classes[batch, : i + 1] = classes[: i + 1]

    return out_scores, out_boxes, out_classes
