"""
Evaluation metrics for SpatialMQA.
"""

from copy import deepcopy
from typing import Any, Dict, List

from ..constants import SPATIAL_RELATIONS
from ..datasets.preprocessing import extract_predicted_relation, match_prediction


def _empty_metrics() -> Dict[str, float]:
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy_x": 0.0,
        "accuracy_y": 0.0,
        "accuracy_z": 0.0,
    }


def _prepare_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply original-repo post-processing before metric calculation."""
    prepared = []
    for item in predictions:
        record = deepcopy(item)
        if record.get("result") == 1:
            record["output"] = record.get("answer", record.get("output", ""))
        prepared.append(record)
    return prepared


def calculate_spatial_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate macro Precision, Recall, F1, and Accuracy for spatial relations.
    """
    total_samples = len(predictions)
    if total_samples == 0:
        return _empty_metrics()

    prepared = _prepare_predictions(predictions)
    counts = {relation: {"pre": 0, "label": 0, "tp": 0} for relation in SPATIAL_RELATIONS}

    for data in prepared:
        answer = data.get("answer", "").lower()
        options = data.get("options")
        output = extract_predicted_relation(data.get("output", ""), options)

        for relation in SPATIAL_RELATIONS:
            if relation in output:
                counts[relation]["pre"] += 1
                break

        for relation in SPATIAL_RELATIONS:
            if relation == answer:
                counts[relation]["label"] += 1
                break

        if match_prediction(output, answer, options):
            for relation in SPATIAL_RELATIONS:
                if relation == answer:
                    counts[relation]["tp"] += 1
                    break

    precision_sum = 0.0
    recall_sum = 0.0
    total_tp = 0

    for relation in SPATIAL_RELATIONS:
        class_counts = counts[relation]
        pre_num = class_counts["pre"] if class_counts["pre"] > 0 else 1
        label_num = class_counts["label"] if class_counts["label"] > 0 else 1
        precision_sum += class_counts["tp"] / pre_num
        recall_sum += class_counts["tp"] / label_num
        total_tp += class_counts["tp"]

    num_classes = len(SPATIAL_RELATIONS)
    macro_p = precision_sum / num_classes
    macro_r = recall_sum / num_classes
    f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    accuracy = total_tp / total_samples if total_samples > 0 else 0.0

    position_groups = {
        "y": ["on/above", "below"],
        "z": ["in front of", "behind"],
        "x": ["left of", "right of"],
    }

    xyz_accuracies = {}
    for axis, relations in position_groups.items():
        axis_total = 0
        axis_tp = 0
        for data in prepared:
            answer = data.get("answer", "").lower()
            options = data.get("options")
            output = extract_predicted_relation(data.get("output", ""), options)
            if answer in relations:
                axis_total += 1
                if match_prediction(output, answer, options):
                    axis_tp += 1
        xyz_accuracies[axis] = axis_tp / axis_total if axis_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_p,
        "recall": macro_r,
        "f1": f1,
        "accuracy_x": xyz_accuracies["x"],
        "accuracy_y": xyz_accuracies["y"],
        "accuracy_z": xyz_accuracies["z"],
    }
