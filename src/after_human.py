import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import json
from paddleocr import PaddleOCR
from transformers import AutoModelForObjectDetection
from utils import parse_xml, compute_calibration_scores
from tsr_ocr import extract_tsr_ocr_confidences
from score_functions import aps_conformal_score  
from after_uq import compute_after_uq_metrics
from before_uq import compute_before_uq_metrics
import argparse
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="Path to the images")
    parser.add_argument("--xml_dir", help="Path to the XML files")
    parser.add_argument("--out_dir", help="Path to save the computed metrics")
    parser.add_argument("--viz_dir", help="Path to save the visualizations")
    parser.add_argument("--test_images", help="path to JSON file containing test images")
    parser.add_argument("--domain_name", help="name of the domain (Biology, ICDAR, MatSci, CompSci)")
    args = parser.parse_args()
    return args

# Storage for After Human Correction results
after_human_results = []

### **📌 Compute Metrics After Human Correction (Using Distance-Based Matching)**
def compute_after_human_correction(image_path, ground_truth_path, flagged_incorrect, viz_dir, tsr_model, ocr_model):
    """
    Compute precision, recall, and error rate assuming humans correct only flagged incorrect extractions.
    This version uses a distance-based approach to find the closest matching ground truth cell.
    """
    # ✅ Step 1: Load Ground Truth Cells
    ground_truth_cells = parse_xml(ground_truth_path)

    # ✅ Step 2: Load Extracted Cells (Predictions)
    extracted_cells = extract_tsr_ocr_confidences(image_path, tsr_model, ocr_model)

    # ✅ Step 3: Convert Ground Truth Cells into a Set for Fast Lookup
    gt_positions = [(gt["start_row"], gt["start_col"], gt["end_row"], gt["end_col"], gt["text"].strip()) for gt in ground_truth_cells]

    # ✅ Step 4: Apply Distance-Based Matching for Human Correction
    corrected_extractions = []

    for cell in extracted_cells:
        cell_position = (cell["start_row"], cell["start_col"], cell["end_row"], cell["end_col"])
        
        # If cell is flagged as incorrect, find best matching ground truth cell using position distance
        if cell_position in {(f["start_row"], f["start_col"], f["end_row"], f["end_col"]) for f in flagged_incorrect}:
            best_match = find_closest_gt_cell(cell, ground_truth_cells)

            # If a match is found, apply correction
            if best_match:
                cell["text"] = best_match["text"]
                cell["start_row"] = best_match["start_row"]
                cell["start_col"] = best_match["start_col"]
                cell["end_row"] = best_match["end_row"]
                cell["end_col"] = best_match["end_col"]

        corrected_extractions.append(cell)  # Store corrected prediction

    # ✅ Step 5: Identify Remaining Incorrect Cells After Human Correction
    remaining_incorrect = []
    gt_tuples = {
        (gt["start_row"], gt["start_col"], gt["end_row"], gt["end_col"], gt["text"].strip())
        for gt in ground_truth_cells
    }

    for cell in extracted_cells:
        cell_tuple = (cell["start_row"], cell["start_col"], cell["end_row"], cell["end_col"], cell["text"].strip())

        if cell_tuple not in gt_tuples:
            remaining_incorrect.append(cell)  # Still incorrect

    # ✅ Step 6: Compute Metrics After Human Correction
    total_extracted = len(extracted_cells)
    incorrect_count_after_human = len(remaining_incorrect)
    correct_count_after_human = total_extracted - incorrect_count_after_human

    # Compute precision as proportion of correct extractions
    data_accuracy = correct_count_after_human / total_extracted if total_extracted > 0 else 0
    # Compute error rate as proportion of incorrect extractions
    error_rate_after = incorrect_count_after_human / total_extracted if total_extracted > 0 else 0

    # ✅ Store results
    after_human_results.append({
        "image_path": image_path,
        "data_accuracy_after_human": data_accuracy,
        "error_rate_after_human": error_rate_after
    })

    # ✅ Debugging Output
    print(f"📝 Image: {image_path}")
    print(f"🔴 Total Extracted Cells: {total_extracted}")
    print(f"⚠️ Remaining Incorrect Extractions After Human Correction: {incorrect_count_after_human}")
    print(f"✅ Data Accuracy: {data_accuracy:.4f},  Error Rate: {error_rate_after:.4f}")

    # ✅ Generate visualization for remaining incorrect extractions
    highlight_remaining_incorrect(image_path, remaining_incorrect, viz_dir)

    return remaining_incorrect


### **📌 Step 4: Function to Find Closest Matching Ground Truth Cell**
def find_closest_gt_cell(pred_cell, ground_truth_cells):
    """
    Finds the closest ground truth cell to the predicted cell using position-based distance.
    """
    min_distance = float("inf")
    best_match = None

    pred_position = np.array([pred_cell["start_row"], pred_cell["start_col"], pred_cell["end_row"], pred_cell["end_col"]])

    for gt in ground_truth_cells:
        gt_position = np.array([gt["start_row"], gt["start_col"], gt["end_row"], gt["end_col"]])

        # Compute Euclidean distance
        distance = np.linalg.norm(pred_position - gt_position)

        # Update best match if distance is smaller
        if distance < min_distance:
            min_distance = distance
            best_match = gt

    return best_match  # Return the closest matching ground truth cell


# **🖼️ Visualization of Remaining Incorrect Extractions**
def highlight_remaining_incorrect(image_path, remaining_incorrect, viz_dir):
    """
    Generate and save a visualization highlighting remaining incorrect extractions after human correction.
    """
    # Load image
    img = cv2.imread(image_path)

    # Draw bounding boxes
    for incorrect_cell in remaining_incorrect:
        x1, y1, x2, y2 = incorrect_cell['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

    # Save visualization
    save_path = os.path.join(viz_dir, os.path.basename(image_path).replace('.png', '_remaining_incorrect.png'))
    cv2.imwrite(save_path, img)

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
    # Load test images from JSON
    domains_test_data = load_test_images(test_images_file_path)[args.domain_name]["test_data"]
    # Process all test images
    test_images = [item['image_name'] for item in domains_test_data]
    best_thresholds = [item['best_threshold'] for item in domains_test_data]

    # Initialize TSR & OCR models
    tsr_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    cal_paths = [os.path.join(args.img_dir, img) for img in os.listdir(args.img_dir) if os.path.join(args.img_dir, img) not in test_images]
    # Compute the calibration scores
    calibration_data = [extract_tsr_ocr_confidences(path, tsr_model, ocr_model) for path in cal_paths]
    calibration_scores_aps = compute_calibration_scores(calibration_data, aps_conformal_score)
    print("=====================Done Computing Calibration Scores =================")

    for idx, img_name in enumerate(test_images):
        image_path = os.path.join(args.img_dir, img_name)
        best_threshold = best_thresholds[idx]
        print(f"=====================Processing {image_path}=========================")
        base_name = os.path.basename(image_path)    #(row["image_name"])
        gt_path = os.path.join(xml_dir, base_name[:-3] + "xml")  # Assuming ground truth paths are stored in the CSV
        incorrect_extractions_before_uq = compute_before_uq_metrics(image_path, gt_path, viz_dir, 
                                                                    ocr_model, tsr_model)
        flagged_incorect = compute_after_uq_metrics(image_path, gt_path, 
                                                    incorrect_extractions_before_uq,
                                                    calibration_scores_aps, best_threshold, viz_dir, 
                                                    aps_conformal_score, tsr_model, ocr_model)
        #print(f"==============Flagged Incorrect results: \n{flagged_incorect}================")
        remaining_incorrect = compute_after_human_correction(image_path, gt_path, flagged_incorect, 
                                                             viz_dir, tsr_model, ocr_model)
        #print(f"==============Remaining Incorrect results: \n{remaining_incorrect}================")
    # Save the results
    after_human_df = pd.DataFrame(after_human_results)
    after_human_df.to_csv(os.path.join(out_dir, "after_human_" + args.domain_name + ".csv"), index=False)

    # Display summary
    print("========== After Human Correction Evaluation ==========")
    print(after_human_df.describe())
