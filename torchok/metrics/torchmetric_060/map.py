# Copyright The PyTorch Lightning team.
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
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torchmetrics import Metric
from torchvision.ops import box_convert

from torchok.constructor import METRICS

log = logging.getLogger(__name__)


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""

    map: Tensor
    map_50: Tensor
    map_75: Tensor
    map_small: Tensor
    map_medium: Tensor
    map_large: Tensor
    mar_1: Tensor
    mar_10: Tensor
    mar_100: Tensor
    mar_small: Tensor
    mar_medium: Tensor
    mar_large: Tensor
    map_per_class: Tensor
    mar_100_per_class: Tensor

    def __getitem__(self, key: str) -> Union[Tensor, List[Tensor]]:
        return getattr(self, key)


# noinspection PyMethodMayBeStatic
class WriteToLog:
    """Logging class to move logs to log.debug()."""

    def write(self, buf: str) -> None:  # skipcq: PY-D0003, PYL-R0201
        for line in buf.rstrip().splitlines():
            log.debug(line.rstrip())

    def flush(self) -> None:  # skipcq: PY-D0003, PYL-R0201
        for handler in log.handlers:
            handler.flush()

    def close(self) -> None:  # skipcq: PY-D0003, PYL-R0201
        for handler in log.handlers:
            handler.close()


class _hide_prints:
    """Internal helper context to suppress the default output of the pycocotools package."""

    def __init__(self) -> None:
        self._original_stdout = None

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout  # type: ignore
        sys.stdout = WriteToLog()  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        sys.stdout.close()
        sys.stdout = self._original_stdout  # type: ignore


def _input_validator(preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
    """Ensure the correct input format of `preds` and `targets`"""

    if not isinstance(preds, Sequence):
        raise ValueError("Expected argument `preds` to be of type List")
    if not isinstance(targets, Sequence):
        raise ValueError("Expected argument `target` to be of type List")
    if len(preds) != len(targets):
        raise ValueError("Expected argument `preds` and `target` to have the same length")

    for k in ["boxes", "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in ["boxes", "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(pred["boxes"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all boxes in `preds` to be of type torch.Tensor")
    if any(type(pred["scores"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type torch.Tensor")
    if any(type(pred["labels"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type torch.Tensor")
    if any(type(target["boxes"]) is not torch.Tensor for target in targets):
        raise ValueError("Expected all boxes in `target` to be of type torch.Tensor")
    if any(type(target["labels"]) is not torch.Tensor for target in targets):
        raise ValueError("Expected all labels in `target` to be of type torch.Tensor")

    for i, item in enumerate(targets):
        if item["boxes"].size(0) != item["labels"].size(0):
            raise ValueError(
                f"Input boxes and labels of sample {i} in targets have a"
                f" different length (expected {item['boxes'].size(0)} labels, got {item['labels'].size(0)})"
            )
    for i, item in enumerate(preds):
        if item["boxes"].size(0) != item["labels"].size(0) != item["scores"].size(0):
            raise ValueError(
                f"Input boxes, labels and scores of sample {i} in preds have a"
                f" different length (expected {item['boxes'].size(0)} labels and scores,"
                f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})"
            )


@METRICS.register_class
class CocoEvalMAP(Metric):
    r"""
    Computes the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)\
    <https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173>`_\
    for object detection predictions.
    Optionally, the mAP and mAR values can be calculated per class.
    Predicted boxes and targets have to be in Pascal VOC format
    (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    See the :meth:`update` method for more information about the input format to this metric.
    .. note::
        This metric is a wrapper for the
        `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        which is a standard implementation for the mAP metric for object detection. Using this metric
        therefore requires you to have `pycocotools` installed. Please install with ``pip install pycocotools`` or
        ``pip install torchmetrics[detection]``.
    .. note::
        This metric requires you to have `torchvision` version 0.8.0 or newer installed (with corresponding
        version 1.7.0 of torch or newer). Please install with ``pip install torchvision`` or
        ``pip install torchmetrics[detection]``.
    .. note::
        As the pycocotools library cannot deal with tensors directly, all results have to be transferred
        to the CPU, this might have a performance impact on your training.
    Args:
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact. default: False
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather
        displayed_metrics:
            List of metrics that will be returned. May contain the following values:
                - map: ``torch.Tensor``
                - map_50: ``torch.Tensor``
                - map_75: ``torch.Tensor``
                - map_small: ``torch.Tensor``
                - map_medium: ``torch.Tensor``
                - map_large: ``torch.Tensor``
                - mar_1: ``torch.Tensor``
                - mar_10: ``torch.Tensor``
                - mar_100: ``torch.Tensor``
                - mar_small: ``torch.Tensor``
                - mar_medium: ``torch.Tensor``
                - mar_large: ``torch.Tensor``
                - map_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
                - mar_100_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
    Raises:
        ImportError:
            If ``pycocotools`` is not installed
        ImportError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ValueError:
            If ``class_metrics`` is not a boolean
    """

    def __init__(
            self,
            class_metrics: bool = False,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
            displayed_metrics: Optional[List[str]] = None
    ) -> None:  # type: ignore
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics
        if displayed_metrics is None:
            displayed_metrics = {
                "map", "map_50", "map_75", "map_small",
                "map_medium", "map_large", "mar_1", "mar_10",
                "mar_100", "mar_small", "mar_medium", "mar_large",
                "map_per_class", "mar_100_per_class"
            }
        self.displayed_metrics = displayed_metrics

        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        super(CocoEvalMAP, self).reset()
        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and ground truth to the metric.
        Args:
            preds: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):
                - ``boxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 5]`` containing ``num_boxes`` detection boxes
                  of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax, score]`` in absolute image coordinates.
                - ``labels``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 0-indexed detection classes
                  for the boxes.
                - ``masks``: ``torch.bool`` of shape ``[num_boxes, image_height, image_width]`` containing boolean
                  masks. Only required when `iou_type="segm"`.
            target: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):
                - ``bboxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 4]`` containing ``num_boxes``
                  ground truth boxes of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax]`` in absolute image coordinates.
                - ``label``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 0-indexed ground truth
                   classes for the boxes.
                - ``masks``: ``torch.bool`` of shape ``[num_boxes, image_height, image_width]`` containing boolean
                  masks. Only required when `iou_type="segm"`.
        Raises:
            ValueError:
                If ``preds`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``target`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        for i in range(len(preds)):
            bboxes_with_scores = preds[i].pop('bboxes')
            preds[i]['boxes'] = bboxes_with_scores[:, :4]
            preds[i]['scores'] = bboxes_with_scores[:, 4]

        for i in range(len(target)):
            target[i]['boxes'] = target[i].pop('bboxes')

        _input_validator(preds, target)

        for item in preds:
            self.detection_boxes.append(item["boxes"].cpu().detach().float())
            self.detection_scores.append(item["scores"].cpu().detach().float())
            self.detection_labels.append(item["labels"].cpu().detach().long())

        for item in target:
            self.groundtruth_boxes.append(item["boxes"].cpu().detach())
            self.groundtruth_labels.append(item["labels"].cpu().detach())

    def compute(self) -> dict:
        """Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` scores. All detections added in
        the `update()` method are included.
        Note:
            Main `map` score is calculated with @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
        Returns:
            dict containing
            - map: ``torch.Tensor``
            - map_50: ``torch.Tensor``
            - map_75: ``torch.Tensor``
            - map_small: ``torch.Tensor``
            - map_medium: ``torch.Tensor``
            - map_large: ``torch.Tensor``
            - mar_1: ``torch.Tensor``
            - mar_10: ``torch.Tensor``
            - mar_100: ``torch.Tensor``
            - mar_small: ``torch.Tensor``
            - mar_medium: ``torch.Tensor``
            - mar_large: ``torch.Tensor``
            - map_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
            - mar_100_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
        """
        coco_target, coco_preds = COCO(), COCO()
        coco_target.dataset = self._get_coco_format(self.groundtruth_boxes, self.groundtruth_labels)
        coco_preds.dataset = self._get_coco_format(self.detection_boxes, self.detection_labels, self.detection_scores)

        with _hide_prints():
            coco_target.createIndex()
            coco_preds.createIndex()
            coco_eval = COCOeval(coco_target, coco_preds, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        map_per_class_values: Tensor = torch.Tensor([-1])
        mar_100_per_class_values: Tensor = torch.Tensor([-1])
        # if class mode is enabled, evaluate metrics per class
        if self.class_metrics:
            map_per_class_list = []
            mar_100_per_class_list = []
            for class_id in torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist():
                coco_eval.params.catIds = [class_id]
                with _hide_prints():
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    class_stats = coco_eval.stats

                map_per_class_list.append(torch.Tensor([class_stats[0]]))
                mar_100_per_class_list.append(torch.Tensor([class_stats[8]]))
            map_per_class_values = torch.Tensor(map_per_class_list)
            mar_100_per_class_values = torch.Tensor(mar_100_per_class_list)

        metrics = MAPMetricResults(
            map=torch.Tensor([stats[0]]),
            map_50=torch.Tensor([stats[1]]),
            map_75=torch.Tensor([stats[2]]),
            map_small=torch.Tensor([stats[3]]),
            map_medium=torch.Tensor([stats[4]]),
            map_large=torch.Tensor([stats[5]]),
            mar_1=torch.Tensor([stats[6]]),
            mar_10=torch.Tensor([stats[7]]),
            mar_100=torch.Tensor([stats[8]]),
            mar_small=torch.Tensor([stats[9]]),
            mar_medium=torch.Tensor([stats[10]]),
            mar_large=torch.Tensor([stats[11]]),
            map_per_class=map_per_class_values,
            mar_100_per_class=mar_100_per_class_values,
        )

        return {v: metrics.__dict__[v] for v in self.displayed_metrics}

    def _get_coco_format(
            self, boxes: List[torch.Tensor], labels: List[torch.Tensor], scores: Optional[List[torch.Tensor]] = None
    ) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.
        Format is defined at https://cocodataset.org/#format-data
        """

        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        boxes = [box_convert(box, in_fmt="xyxy", out_fmt="xywh") for box in boxes]
        for image_id, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
            image_boxes = image_boxes.cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({"id": image_id})
            for k, (image_box, image_label) in enumerate(zip(image_boxes, image_labels)):
                if len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if type(image_label) != int:
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": image_box,
                    "category_id": image_label,
                    "area": image_box[2] * image_box[3],
                    "iscrowd": 0,
                }
                if scores is not None:
                    score = scores[image_id][k].cpu().tolist()
                    if type(score) != float:
                        raise ValueError(
                            f"Invalid input score of sample {image_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [
            {"id": i, "name": str(i)}
            for i in torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist()
        ]
        return {"images": images, "annotations": annotations, "categories": classes}
