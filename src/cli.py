#!/usr/bin/env python3
"""
CLI module for tag prediction on algorithmic problems.
"""

import argparse
import sys
import json
import pandas as pd

from src.config import FOCUS_TAGS
from src.utils import parse_tags
from src.evaluation import compute_metrics, print_metrics, get_classification_report


def get_predictor(model_name):
    """
    Factory function to get the appropriate predictor.
    """
    if model_name == "codebert":
        from src.predict import CodeBertPredictor
        return CodeBertPredictor()
    elif model_name == "retrieval":
        from src.predict import RetrievalPredictor
        return RetrievalPredictor()
    elif model_name == "llm":
        from src.predict import LLMPredictor
        return LLMPredictor()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from 'codebert', 'retrieval', 'llm'.")


def get_input_from_file(file_path):
    """
    Get problem description and code from a JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    description = data.get("description", data.get("description_clean", ""))
    code = data.get("code", data.get("code_clean", ""))

    return description, code


def cmd_predict(args):
    """
    Handle the predict command for single prediction.
    """
    # Get input from different sources
    if args.input_file:
        print(f"Loading input from {args.input_file}...")
        description, code = get_input_from_file(args.input_file)
    elif args.description:
        description = args.description
        code = args.code or ""
    else:
        print("Error: Please provide either --input-file or --description")
        sys.exit(1)

    if not description.strip():
        print("Error: Description cannot be empty.")
        sys.exit(1)

    print(f"\nLoading {args.model} model...")
    predictor = get_predictor(args.model)

    # Prepare input
    data = [{
        "description_clean": description,
        "code_clean": code,
    }]

    # Run prediction
    print("Running prediction...")
    if args.model == "codebert":
        predictions, inference_time = predictor.predict(data, threshold=0.5)
    elif args.model == "retrieval":
        predictions, inference_time = predictor.predict(data, k=3)
    elif args.model == "llm":
        predictions, inference_time = predictor.predict(data, include_code=bool(code))

    predicted_tags = predictions[0]
    print("Prediction Result")
    print(f"Predicted tags: {predicted_tags}")
    print(f"Inference time: {inference_time:.3f}s")

    return predicted_tags


def cmd_evaluate(args):
    """
    Handle the evaluate command for dataset evaluation.
    """
    print(f"Loading test data from {args.test_file}...")
    test_df = pd.read_csv(args.test_file)

    data = test_df.to_dict("records")
    y_true = [parse_tags(row.get("tags_filtered", [])) for row in data]

    models_to_evaluate = ["codebert", "retrieval", "llm"] if args.model == "all" else [args.model]

    results = {}

    for model_name in models_to_evaluate:
        print(f"Evaluating {model_name}...")

        try:
            predictor = get_predictor(model_name)

            if model_name == "codebert":
                y_pred, inference_time = predictor.predict(data, threshold=0.5)
            elif model_name == "retrieval":
                y_pred, inference_time = predictor.predict(data, k=3)
            elif model_name == "llm":
                y_pred, inference_time = predictor.predict(data, include_code=True)

            print(f"Total inference time: {inference_time:.2f}s")

            metrics = compute_metrics(y_true, y_pred)
            print_metrics(metrics, model_name=model_name)

            print("\nClassification Report:")
            print(get_classification_report(y_true, y_pred))

            results[model_name] = metrics

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLI for tag prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict tags for a single problem")
    predict_parser.add_argument(
        "--model",
        type=str,
        choices=["codebert", "retrieval", "llm"],
        default="codebert",
        help="Model to use (default: codebert)",
    )
    predict_parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to JSON file with description and code",
    )
    predict_parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Problem description text",
    )
    predict_parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Solution code",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model on test dataset")
    evaluate_parser.add_argument(
        "--model",
        type=str,
        choices=["codebert", "retrieval", "llm", "all"],
        default="codebert",
        help="Model to evaluate (default: codebert)",
    )
    evaluate_parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test CSV file",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
