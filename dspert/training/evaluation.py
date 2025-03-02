import logging
from typing import List, Tuple
from dspert.dataset import Dataset
logger = logging.getLogger(__name__)

def precision_recall_f1_report(y_gold: List[List[Tuple[str, int, int]]], y_pred: List[List[Tuple[str, int, int]]]) -> Tuple[float, float, float]:
    """
    Calculate micro-average precision, recall, and F1-score for entity recognition.

    Parameters
    ----------
    y_gold : List[List[Tuple[str, int, int]]]
        Ground truth chunks.
    y_pred : List[List[Tuple[str, int, int]]]
        Predicted chunks.

    Returns
    -------
    Tuple[float, float, float]
        Precision, Recall, F1 scores (micro-average).
    """
    n_gold = sum(len(sample) for sample in y_gold)
    n_pred = sum(len(sample) for sample in y_pred)
    n_true_positive = sum(len(set(g) & set(p)) for g, p in zip(y_gold, y_pred))

    precision = n_true_positive / n_pred if n_pred > 0 else 0.0
    recall = n_true_positive / n_gold if n_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def _disp_prf(precision: float, recall: float, f1: float):
    """Display precision, recall, and F1 scores."""
    logger.info(f"Micro Precision: {precision*100:.3f}%")
    logger.info(f"Micro Recall: {recall*100:.3f}%")
    logger.info(f"Micro F1-score: {f1*100:.3f}%")

def evaluate_entity_recognition(trainer, dataset: Dataset, batch_size: int = 32, save_preds: bool = False):
    """
    Evaluate entity recognition results using SimpleTrainer and Dataset.

    Parameters
    ----------
    trainer : SimpleTrainer
        The trainer instance with the model to evaluate.
    dataset : Dataset
        The dataset containing samples to evaluate or predict.
    batch_size : int, optional
        Batch size for prediction (default: 32).
    save_preds : bool, optional
        Save predictions into dataset.data if True (default: False).
    """
    # Dự đoán chunks
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)

    if save_preds:
        # Lưu dự đoán vào dataset.data
        for ex, chunks_pred in zip(dataset.data, set_y_pred):
            ex['chunks_pred'] = chunks_pred
        logger.info("ER | Predictions saved")
    else:
        # Lấy ground truth từ dataset
        set_y_gold = [ex['chunks'] for ex in dataset.data]
        
        # Tính điểm
        precision, recall, f1 = precision_recall_f1_report(set_y_gold, set_y_pred)
        _disp_prf(precision, recall, f1)