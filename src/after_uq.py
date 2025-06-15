import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from paddleocr import PaddleOCR
from transformers import AutoModelForObjectDetection
from utils import parse_xml, match_cells
from tsr_ocr import extract_tsr_ocr_confidences
from utils import compute_calibration_scores
from score_functions import aps_conformal_score, hybrid_spatial_score
import argparse
import cv2
import json
from src.before_uq import compute_before_uq_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="Path to the images")
    parser.add_argument("--xml_dir", help="Path to the xml files")
    parser.add_argument("--out_dir", help="path to save the computed metrics")
    parser.add_argument("--viz_dir", help="path to save the visualizations")
    parser.add_argument("--test_images", help="path to JSON file containing test images")
    parser.add_argument("--domain_name", help="name of the domain name (Biology, ICDAR, MatSci, CompSci)")
    args = parser.parse_args()
    return args


# Storage for After UQ results
after_uq_results = []

def compute_labor_savings(total_extracted_cells, flagged_cells):
    """
    Compute labor savings due to UQ by measuring how much manual effort is reduced.
    
    Parameters:
    - image_path (str): Path to the image being processed.
    - total_extracted_cells (int): Total number of extracted cells (without UQ).
    - flagged_cells (int): Number of flagged cells that require human correction.

    Returns:
    - labor_savings (float): Proportion of labor saved due to UQ.
    """
    if total_extracted_cells == 0:
        return 0  # Avoid division by zero
    
    labor_savings = (total_extracted_cells - flagged_cells) / total_extracted_cells

    # ✅ Debugging Output
    print(f"🔢 Total Extracted Cells: {total_extracted_cells}")
    print(f"🚩 Flagged Cells (Needing Review): {flagged_cells}")
    print(f"💼 Labor Savings Due to UQ: {labor_savings:.4f}")

    return labor_savings



def compute_after_uq_metrics(image_path, incorrect_extractions_before_uq, cal_scores, 
                             best_threshold, viz_dir, score_fn, tsr_model, ocr_model, alpha=0.1):
    """
    Compute precision, recall, and F1-score after applying Uncertainty Quantification (UQ).
    Instead of matching extracted cells with ground truth, we link it with the incorrect extractions from Before UQ.
    """
    # Calculate the threshold using the alpha value
    n = len(cal_scores)  # Number of calibration samples
    q_level = np.ceil((n + 1) * (1 - alpha)) / n  # Adjusted quantile level
    threshold = np.quantile(cal_scores, q_level, method = "higher")
    # ✅ Step 1: Extract Table Data using TSR & OCR
    extracted_cells = extract_tsr_ocr_confidences(image_path, tsr_model, ocr_model)

    # ✅ Step 2: Compute Uncertainty Score using APS
    for cell in extracted_cells:
        tsr_conf = cell["tsr_combined_conf"]
        ocr_conf = cell["ocr_confidence"] if cell["ocr_confidence"] is not None else tsr_conf
        if isinstance(tsr_conf, torch.Tensor):
            tsr_conf = tsr_conf.item()
        if isinstance(ocr_conf, torch.Tensor):
            ocr_conf = ocr_conf.item()
        score, _, _ = score_fn(tsr_conf, ocr_conf)
        score = float(score)
        uncertainty_score = min(1.0, abs(score - threshold) / threshold)
        cell["uncertainty_score"] = uncertainty_score

    # ✅ Step 3: Flag High-Uncertainty Cells
    flagged_incorrect = [cell for cell in extracted_cells if cell["uncertainty_score"] > best_threshold]

    # ✅ Step 4: Compute UQ Metrics using Before UQ results
    flagged_cell_tuples = {
        (cell["start_row"], cell["start_col"], cell["end_row"], cell["end_col"], cell["text"].strip()) for cell in flagged_incorrect
    }
    
    incorrect_cell_tuples_before_uq = {
        (cell["start_row"], cell["start_col"], cell["end_row"], cell["end_col"], cell["text"].strip()) for cell in incorrect_extractions_before_uq
    }

    # Compute how many flagged extractions were actually incorrect
    correct_flags = len(flagged_cell_tuples.intersection(incorrect_cell_tuples_before_uq))
    incorrect_flags = len(flagged_cell_tuples - incorrect_cell_tuples_before_uq)  # False flags
    total_incorrect_before_uq = len(incorrect_cell_tuples_before_uq)

    # ✅ Precision and Recall of UQ
    precision_uq = correct_flags / (correct_flags + incorrect_flags) if (correct_flags + incorrect_flags) > 0 else 0
    recall_uq = correct_flags / total_incorrect_before_uq if total_incorrect_before_uq > 0 else 0
    f1_uq = (2 * precision_uq * recall_uq) / (precision_uq + recall_uq) if (precision_uq + recall_uq) > 0 else 0

    # ✅ Debugging Output
    print(f"📝 Image: {image_path}")
    print(f"⚠️ Flagged Incorrect Extractions: {len(flagged_incorrect)}")
    print(f"✅ Correctly Flagged Errors: {correct_flags}")
    print(f"❌ False Flags (Correct Cells Wrongly Flagged): {incorrect_flags}")
    print(f"📊 Precision (After UQ): {precision_uq:.4f}, Recall: {recall_uq:.4f}, F1: {f1_uq:.4f}")

    # Compute Labor Savings
    labor_savings = compute_labor_savings(len(extracted_cells), len(flagged_incorrect))

    # ✅ Store results
    after_uq_results.append({
        "image_path": image_path,
        "precision_after_uq": precision_uq,
        "recall_after_uq": recall_uq,
        "f1_after_uq": f1_uq,
        "labor_savings": labor_savings
    })

    # ✅ Generate visualization of flagged incorrect extractions
    highlight_flagged_extractions(image_path, flagged_incorrect, viz_dir)

    return flagged_incorrect  # Return flagged cells for After Human Correction


def highlight_flagged_extractions(image_path, flagged_incorrect, viz_dir):
    """
    Generate and save a visualization highlighting remaining incorrect extractions **after human correction** using OpenCV.
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)

    # Convert image to RGB (OpenCV loads images in BGR format)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    for incorrect_pred in flagged_incorrect:
        x1, y1, x2, y2 = incorrect_pred['bbox']  # Extract bbox coordinates
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 225), 2)  # Draw red box (BGR: Blue, Green, Red)

    # Save the new image with bounding boxes
    save_path = os.path.join(viz_dir, os.path.basename(image_path).replace('.png', '_flagged_incorrect.png'))
    cv2.imwrite(save_path, img)  # Convert back to BGR before saving

def load_test_images(test_images_file_path):
    with open(test_images_file_path, 'r') as f:
        domains = json.load(f)

    return domains


if __name__ == '__main__':
    args = get_args()
    xml_dir = args.xml_dir
    out_dir = args.out_dir
    viz_dir = args.viz_dir
    test_images_file_path = args.test_images
    # Initialize TSR & OCR models
    tsr_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    # Load test images from JSON
    domains_test_data = load_test_images(test_images_file_path)[args.domain_name]["test_data"]
    # Process all test images
    test_images = [item['image_name'] for item in domains_test_data]
    best_thresholds = [item['best_threshold'] for item in domains_test_data]
    cal_paths = [os.path.join(args.img_dir, img) for img in os.listdir(args.img_dir) if os.path.join(args.img_dir, img) not in test_images]
    # Compute the calibration scores
    calibration_data = [extract_tsr_ocr_confidences(path, tsr_model, ocr_model) for path in cal_paths]
    calibration_scores_aps = compute_calibration_scores(calibration_data, aps_conformal_score)  # using APS conformal score
    print("=====================Done Computing Calibration Scores =================")

    for idx, img_name in enumerate(test_images):
        image_path = os.path.join(args.img_dir, img_name)
        best_threshold = best_thresholds[idx]
        print(f"=====================Processing {image_path}=========================")
        base_name = os.path.basename(image_path)    #(row["image_name"])
        gt_path = os.path.join(xml_dir, base_name[:-3] + "xml")  # Assuming ground truth paths are stored in the CSV
        incorrect_extractions_before_uq = compute_before_uq_metrics(image_path, gt_path, viz_dir, 
                                                                    ocr_model, tsr_model)
        compute_after_uq_metrics(image_path, incorrect_extractions_before_uq, 
                                 calibration_scores_aps, best_threshold, viz_dir, 
                                 aps_conformal_score, tsr_model, ocr_model)

    # Save the results
    after_uq_df = pd.DataFrame(after_uq_results)
    after_uq_df.to_csv(os.path.join(out_dir, "after_uq_" + args.domain_name + ".csv"), index=False)

    # Display summary
    print("========== After UQ Evaluation ==========")
    print(after_uq_df.describe())

