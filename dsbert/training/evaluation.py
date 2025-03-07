import logging
from typing import List, Tuple
from dsbert.dataset import Dataset
from collections import defaultdict

logger = logging.getLogger(__name__)

def precision_recall_f1_report(y_gold: List[List[Tuple[str, int, int]]], y_pred: List[List[Tuple[str, int, int]]]) -> Tuple:
    """
    Calculate precision, recall, and F1 score for entity recognition.

    Args:
        y_gold (List[List[Tuple[str, int, int]]]): Gold standard chunks.
        y_pred (List[List[Tuple[str, int, int]]]): Predicted chunks.

    Returns:
        Tuple: (micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, per_label_metrics)
            micro_p, micro_r, micro_f1: micro-averaged precision, recall, and F1 score.
            macro_p, macro_r, macro_f1: macro-averaged precision, recall, and F1 score.
            per_label_metrics: a dictionary of label to (precision, recall, f1) metrics.
    """
    
    if len(y_gold) != len(y_pred):
        raise ValueError(f"Length mismatch: y_gold ({len(y_gold)}) and y_pred ({len(y_pred)}) must have the same length.")
    
    n_gold, n_pred = sum(map(len, y_gold)), sum(map(len, y_pred))
    n_tp = sum(len(set(g) & set(p)) for g, p in zip(y_gold, y_pred))

    micro_p = n_tp / n_pred if n_pred else 0.0
    micro_r = n_tp / n_gold if n_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    label_stats = defaultdict(lambda: defaultdict(int))

    for g_sample, p_sample in zip(y_gold, y_pred):
        g_set, p_set = set(g_sample), set(p_sample)

        for chunk in g_set & p_set:
            label_stats[chunk[0]]["tp"] += 1
        for chunk in g_set - p_set:
            label_stats[chunk[0]]["fn"] += 1
        for chunk in p_set - g_set:
            label_stats[chunk[0]]["fp"] += 1

    per_label_metrics = {}
    for label, stats in label_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        per_label_metrics[label] = (precision, recall, f1)


    num_labels = len(per_label_metrics)
    macro_p = sum(p for p, _, _ in per_label_metrics.values()) / num_labels if num_labels else 0.0
    macro_r = sum(r for _, r, _ in per_label_metrics.values()) / num_labels if num_labels else 0.0
    macro_f1 = sum(f for _, _, f in per_label_metrics.values()) / num_labels if num_labels else 0.0

    return micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, per_label_metrics

def _disp_prf(metrics: Tuple):
    """Display micro-average, macro-average, and per-label precision, recall, and F1 scores."""
    
    micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, per_label_metrics = metrics
    logger.info("=== Micro-Average Metrics ===")
    logger.info(f"Micro Precision: {micro_p:.4f}")
    logger.info(f"Micro Recall: {micro_r:.4f}")
    logger.info(f"Micro F1-score: {micro_f1:.4f}")
    
    logger.info("=== Macro-Average Metrics ===")
    logger.info(f"Macro Precision: {macro_p:.4f}")
    logger.info(f"Macro Recall: {macro_r:.4f}")
    logger.info(f"Macro F1-score: {macro_f1:.4f}")
    
    logger.info("=== Per-Label Metrics ===")
    for label, (precision, recall, f1) in per_label_metrics.items():
        logger.info(f"Label '{label}': P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

def evaluate_entity_recognition(trainer, dataset: Dataset, batch_size: int = 32, save_preds: bool = False):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)

    if save_preds:
        for ex, chunks_pred in zip(dataset.data, set_y_pred):
            ex["chunks_pred"] = chunks_pred
        logger.info("ER | Predictions saved")
    else:
        set_y_gold = [ex["chunks"] for ex in dataset.data]
        metrics = precision_recall_f1_report(set_y_gold, set_y_pred)
        _disp_prf(metrics)
