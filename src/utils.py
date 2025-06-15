import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import os
from rapidfuzz.distance import Levenshtein

def parse_xml(file_path):
    """
    Parses an XML file and extracts cell details.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    cells = []
    for cell in root.findall('cell'):
        start_row = int(cell.get('start_row'))
        end_row = int(cell.get('end_row'))
        start_col = int(cell.get('start_col'))
        end_col = int(cell.get('end_col'))
        
        # Extract Coords
        coords = cell.find('Coords').get('points') if cell.find('Coords') is not None else None
        
        # Extract text
        text_element = cell.find('text')
        text = text_element.text.strip() if text_element is not None and text_element.text else ''
        
        cells.append({
            'start_row': start_row,
            'end_row': end_row,
            'start_col': start_col,
            'end_col': end_col,
            #'coords': coords,
            'text': text
        })
    return cells


def match_cells(ground_truth_cells, prediction_set):
    """
    Match predicted cells to ground truth cells based on both location and text conformity.
    """
    matches = []
    unmatched_gt = []
    unmatched_pred = prediction_set.copy()

    for gt_cell in ground_truth_cells:
        match_found = False
        for pred_cell in unmatched_pred:
            if (
                int(pred_cell['start_row']) == int(gt_cell['start_row']) and
                int(pred_cell['end_row']) == int(gt_cell['end_row']) and
                int(pred_cell['start_col']) == int(gt_cell['start_col']) and
                int(pred_cell['end_col']) == int(gt_cell['end_col'])
            ):
                matches.append((gt_cell, pred_cell))
                unmatched_pred.remove(pred_cell)
                match_found = True
                break
        if not match_found:
            unmatched_gt.append(gt_cell)

    return matches, unmatched_gt, unmatched_pred

def get_table_texts(matches):
    gt_texts, pred_texts = [], []
    for gt, pred in matches:
        gt_texts.append(gt['text'])
        pred_texts.append(pred['text'])
    return gt_texts, pred_texts

def levenshtein_accuracy(ground_truth_texts, predicted_texts):
    """Computes Levenshtein distance for OCR accuracy"""
    try:
        total_score = 0
        count = min(len(ground_truth_texts), len(predicted_texts))
        for gt, pred in zip(ground_truth_texts, predicted_texts):
            print(f"GT-text: {gt}; Pred-text: {pred}")
            total_score += 1 - (Levenshtein.distance(gt, pred) / max(len(gt), len(pred), 1))
    except Exception as e:
        print(f"Error computing Levenshtein accuracy: {e}")
        return 0
    return total_score / count if count > 0 else 0
def compute_metrics(matches, unmatched_gt, unmatched_pred):
    """
    Computes coverage, text accuracy, and location accuracy metrics.
    
    Returns:
        - coverage (float): Fraction of ground truth cells that have a matching prediction.
        - text_accuracy (float): Fraction of prediction set where the text exactly matches the ground truth.
        - location_accuracy (float): Fraction of prediction set where the location matches the ground truth.
    """
    total_matches = len(matches)
    total_predictions = total_matches + len(unmatched_pred)  # Total predicted cells
    
    # Coverage: Fraction of ground truth cells that have a matching prediction
    coverage = total_matches / (total_matches + len(unmatched_gt)) if (total_matches + len(unmatched_gt)) > 0 else 0
    
    gt_texts, pred_texts = get_table_texts(matches)
    # Text Accuracy: Fraction of correct text matches in the prediction set
    text_accuracy = levenshtein_accuracy(gt_texts, pred_texts)

    # Location Accuracy: Fraction of correct location matches in the prediction set
    correct_location_matches = sum(
        1 for gt, pred in matches 
        if gt['start_row'] == pred['start_row'] and 
           gt['start_col'] == pred['start_col'] and 
           gt['end_row'] == pred['end_row'] and 
           gt['end_col'] == pred['end_col']
    )
    location_accuracy = correct_location_matches / total_predictions if total_predictions > 0 else 0

    return coverage, text_accuracy, location_accuracy


def is_cell_match(predicted_cell, ground_truth_cells):
    """
    Check if a predicted cell matches any ground truth cell.
    """
    for gt_cell in ground_truth_cells:
        if (predicted_cell['start_row'] == gt_cell['start_row'] and
            predicted_cell['end_row'] == gt_cell['end_row'] and
            predicted_cell['start_col'] == gt_cell['start_col'] and
            predicted_cell['end_col'] == gt_cell['end_col'] and
            predicted_cell['text'].strip().lower() == gt_cell['text'].strip().lower()):
            return True
    return False

def compute_correctness(predicted_cells, ground_truth_cells):
    """
    Compute correctness for each predicted cell.
    """
    correctness = []
    for predicted_cell in predicted_cells:
        match = is_cell_match(predicted_cell, ground_truth_cells)
        correctness.append(1 if match else 0)
    return correctness

# Rename cells to match the expected format
def create_prediction(grid_cells):
    for cell in grid_cells:
        cell["start_row"] = cell["row_idx"]
        cell["end_row"] = cell["row_idx"]
        cell["start_col"] = cell["col_idx"]
        cell["end_col"] = cell["col_idx"]
    return grid_cells

# Compute Calibration Scores
def compute_calibration_scores(calibration_data, score_function):
    """
    Compute conformal scores for calibration data.
    """
    scores = []
    for table_data in calibration_data:
        for cell_data in table_data:
            tsr_cell_confidence = cell_data["tsr_combined_conf"]
            ocr_confidence = cell_data["ocr_confidence"] if cell_data["ocr_confidence"] is not None else cell_data["tsr_conf"]
            score, _, _ = score_function(tsr_cell_confidence, ocr_confidence)
            scores.append(score)
    print(f"Scores computed using: {score_function.__name__}: {scores[:10]}")
    return scores

def evaluate_uncertainty_flagging(ground_truth_cells, prediction_set, threshold):
    """
    Evaluate how well the uncertainty score flags incorrect OCR extractions.
    
    Parameters:
        ground_truth_cells (list): List of ground truth cell dictionaries.
        prediction_set (list): List of predicted cell dictionaries.
        threshold (float): The threshold for flagging uncertain extractions.
        score_name (str): The key in prediction_set containing the uncertainty score.

    Returns:
        dict: Precision, recall, and F1 score for uncertainty flagging.
    """
    # Match predictions to ground truth locations
    matched, unmatched_gt, unmatched_pred = match_cells(ground_truth_cells, prediction_set)
    
    TP = 0  # True Positives (Incorrect extractions correctly flagged)
    FP = 0  # False Positives (Correct extractions incorrectly flagged)
    FN = 0  # False Negatives (Incorrect extractions missed)
    try:
        for gt, pred in matched:
            uncertainty_score = pred["uncertainty_score"]  # Extract uncertainty score
            is_flagged = uncertainty_score >= threshold  # Determine if cell is flagged
            is_text_correct = gt['text'].strip() == pred['text'].strip()  # Check correctness
            
            if is_flagged and not is_text_correct:
                TP += 1  # Correctly flagged incorrect text (True Positive)
            elif is_flagged and is_text_correct:
                FP += 1  # Incorrectly flagged correct text (False Positive)
            elif not is_flagged and not is_text_correct:
                FN += 1  # Missed incorrect text (False Negative)

        # Compute precision, recall, and F1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        results = {"precision": precision, "recall": recall, "f1_score": f1_score}
    except Exception as e:
        print(f"Error evaluating uncertainty flagging: {e}")
        return {"precision": 0, "recall": 0, "f1_score": 0}
    return results



def find_best_threshold(ground_truth_cells, prediction_set):
    """
    Find the best uncertainty threshold by maximizing the F1-score.
    
    Parameters:
        ground_truth_cells (list): List of ground truth cell dictionaries.
        prediction_set (list): List of predicted cell dictionaries.
        score_name (str): The key in prediction_set containing the uncertainty score.

    Returns:
        dict: Best threshold and corresponding precision, recall, and F1-score.
    """

    thresholds = np.arange(0.001, 1.0, 0.001)  # Test thresholds from 0.05 to 1.0
    best_threshold = None
    best_f1 = 0
    best_metrics = {}
    try:
        for threshold in thresholds:
            metrics = evaluate_uncertainty_flagging(ground_truth_cells, prediction_set, threshold)
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_threshold = threshold
                best_metrics = metrics  # Save the best precision, recall, and F1
    except Exception as e:
        print(f"Error finding best threshold: {e}")
        return {"best_threshold": None, "precision": 0, "recall": 0, "f1_score": 0}

    return {"best_threshold": best_threshold, **best_metrics}



def process_prediction_sets(ground_truth_cells, prediction_sets_list, score_functions):
    """
    Apply the threshold optimization to each prediction set and save the results in a DataFrame.

    Parameters:
        ground_truth_cells (list): List of ground truth cell dictionaries.
        prediction_sets (list): List of prediction sets for different score functions.
        score_functions (list): List of score function names.

    Returns:
        a list of results containing the best threshold and metrics such as precision, recall, and F1 score
    """
    results = []
    try:
        for score_function, prediction_set in zip(score_functions, prediction_sets_list):
            #score_name = score_names.get(score_function, "uncertainty_score")
            best_threshold_results = find_best_threshold(ground_truth_cells, prediction_set)
            #print(best_threshold_results)

            results.append({
                "score_function": score_function.__name__,
                "best_threshold": best_threshold_results["best_threshold"],
                "precision": round(best_threshold_results["precision"], 5), 
                "recall": round(best_threshold_results["recall"], 5),
                "f1_score": round(best_threshold_results["f1_score"], 5)
            })
        print(results)
    except Exception as e:
        print(f"Error processing prediction sets: {e}")
        return []
    return results


