from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
)

from src.config import FOCUS_TAGS


def compute_metrics(y_true_tags, y_pred_tags, focus_tags=None):
    """
    Compute evaluation metrics for multi-label classification.
    """
    if focus_tags is None:
        focus_tags = FOCUS_TAGS

    # Filter tags to only include focus tags
    y_true_filtered = [[t for t in tags if t in focus_tags] for tags in y_true_tags]
    y_pred_filtered = [[t for t in tags if t in focus_tags] for tags in y_pred_tags]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=focus_tags)
    Y_true = mlb.fit_transform(y_true_filtered)
    Y_pred = mlb.transform(y_pred_filtered)

    # Compute global metrics
    accuracy = accuracy_score(Y_true, Y_pred)
    precision_micro = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)

    precision_macro = precision_score(Y_true, Y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(Y_true, Y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    # Compute per-tag metrics
    per_tag_precision = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    per_tag_recall = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    per_tag_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_tag_support = Y_true.sum(axis=0)

    per_tag_metrics = {}
    for i, tag in enumerate(focus_tags):
        per_tag_metrics[tag] = {
            "precision": per_tag_precision[i],
            "recall": per_tag_recall[i],
            "f1": per_tag_f1[i],
            "support": int(per_tag_support[i]),
        }

    return {
        "accuracy": accuracy,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_tag": per_tag_metrics,
    }


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in a formatted way.
    """
    print(f"Evaluation Results: {model_name}")

    print(f"\nGlobal Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.3f}")
    print(f"  Precision (micro): {metrics['precision_micro']:.3f}")
    print(f"  Recall (micro):    {metrics['recall_micro']:.3f}")
    print(f"  F1 (micro):        {metrics['f1_micro']:.3f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.3f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.3f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.3f}")

    print(f"\nPer-Tag Metrics:")
    print(f"  {'Tag':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")

    for tag, tag_metrics in metrics["per_tag"].items():
        print(
            f"  {tag:<15} {tag_metrics['precision']:>10.3f} {tag_metrics['recall']:>10.3f} "
            f"{tag_metrics['f1']:>10.3f} {tag_metrics['support']:>10}"
        )


def get_classification_report(y_true_tags, y_pred_tags, focus_tags=None):
    """
    Get the sklearn classification report as a string.
    """
    if focus_tags is None:
        focus_tags = FOCUS_TAGS

    # Filter tags to only include focus tags
    y_true_filtered = [[t for t in tags if t in focus_tags] for tags in y_true_tags]
    y_pred_filtered = [[t for t in tags if t in focus_tags] for tags in y_pred_tags]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=focus_tags)
    Y_true = mlb.fit_transform(y_true_filtered)
    Y_pred = mlb.transform(y_pred_filtered)

    return classification_report(Y_true, Y_pred, target_names=focus_tags, zero_division=0)


def get_confusion_matrices(y_true_tags, y_pred_tags, focus_tags=None):
    """
    Get confusion matrix for each tag.
    """
    if focus_tags is None:
        focus_tags = FOCUS_TAGS

    # Filter tags to only include focus tags
    y_true_filtered = [[t for t in tags if t in focus_tags] for tags in y_true_tags]
    y_pred_filtered = [[t for t in tags if t in focus_tags] for tags in y_pred_tags]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=focus_tags)
    Y_true = mlb.fit_transform(y_true_filtered)
    Y_pred = mlb.transform(y_pred_filtered)

    cm = multilabel_confusion_matrix(Y_true, Y_pred)

    result = {}
    for i, tag in enumerate(focus_tags):
        tn, fp, fn, tp = cm[i].ravel()
        result[tag] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }

    return result
