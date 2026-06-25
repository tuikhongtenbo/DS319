"""
Evaluation metrics for SpatialMQA.
"""

from typing import List, Dict, Any
from ..constants import SPATIAL_RELATIONS

def calculate_spatial_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates Precision, Recall, F1, and Accuracy for spatial relations.
    Expects a list of dicts with 'output' (prediction) and 'answer' (ground truth).
    """
    
    # Initialize counts for each relation: [predicted, actual, true_positive]
    counts = {rel: {"pre": 0, "label": 0, "tp": 0} for rel in SPATIAL_RELATIONS}
    
    total_samples = len(predictions)
    if total_samples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for data in predictions:
        output = data.get('output', '').lower()
        answer = data.get('answer', '').lower()
        
        # Match output to one of the relations
        for rel in SPATIAL_RELATIONS:
            if rel in output:
                counts[rel]["pre"] += 1
                break  # Assumes single label output

        # Match answer
        for rel in SPATIAL_RELATIONS:
            if rel == answer:
                counts[rel]["label"] += 1
                break
                
        # Check True Positive
        if answer == output or (answer in output): # Assuming 'output' might have some extra chars but contains answer
            for rel in SPATIAL_RELATIONS:
                if rel == answer:
                    counts[rel]["tp"] += 1
                    break

    # Calculate P, R for each class and average (Macro)
    precision_sum = 0.0
    recall_sum = 0.0
    total_tp = 0
    total_label = 0
    
    for rel in SPATIAL_RELATIONS:
        c = counts[rel]
        pre_num = c["pre"] if c["pre"] > 0 else 1  # Avoid div by zero (following original logic)
        label_num = c["label"] if c["label"] > 0 else 1
        
        precision_sum += c["tp"] / pre_num
        recall_sum += c["tp"] / label_num
        
        total_tp += c["tp"]
        total_label += c["label"]
        
    num_classes = len(SPATIAL_RELATIONS)
    macro_p = precision_sum / num_classes
    macro_r = recall_sum / num_classes
    
    if (macro_p + macro_r) > 0:
        f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
    else:
        f1 = 0.0
        
    accuracy = total_tp / total_samples if total_samples > 0 else 0.0
    
    # Calculate X, Y, Z accuracy
    position_groups = {
        'y': ["on/above", "below"], 
        'z': ["in front of", "behind"],
        'x': ["left of", "right of"]
    }
    
    xyz_accuracies = {}
    for axis, rels in position_groups.items():
        axis_total = 0
        axis_tp = 0
        for data in predictions:
            output = data.get('output', '').lower()
            answer = data.get('answer', '').lower()
            if answer in rels:
                axis_total += 1
                if answer == output or (answer in output):
                    axis_tp += 1
        xyz_accuracies[axis] = axis_tp / axis_total if axis_total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": macro_p,
        "recall": macro_r,
        "f1": f1,
        "accuracy_x": xyz_accuracies['x'],
        "accuracy_y": xyz_accuracies['y'],
        "accuracy_z": xyz_accuracies['z']
    }
