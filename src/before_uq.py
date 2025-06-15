import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from paddleocr import PaddleOCR
from transformers import AutoModelForObjectDetection
from utils import parse_xml
from tsr_ocr import extract_tsr_ocr_confidences  
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", help="Path to save the xml output")
    parser.add_argument("--out_dir", help="path to save the computed metrics")
    parser.add_argument("--viz_dir", help="path to save the computed metrics")
    parser.add_argument("--test_images", help="path to JSON file containing test images")
    parser.add_argument("--domain_name", help="name of the domain (Biology, ICDAR, MatSci, CompSci)")
    args = parser.parse_args()
    return args



# Storage for results
before_uq_results = []

def compute_before_uq_metrics(image_path, ground_truth_path, viz_dir, ocr_model, tsr_model):
    """
    Evaluate TSR-OCR integration by directly comparing extracted cells with ground truth.
    This function also visualizes incorrect extractions by highlighting them.
    """

    # ✅ Step 1: Load Extracted Cells (Predictions)
    extracted_cells = extract_tsr_ocr_confidences(image_path, tsr_model, ocr_model)

    # ✅ Step 2: Load Ground Truth Cells
    ground_truth_cells = parse_xml(ground_truth_path)

    # ✅ Step 3: Convert Ground Truth into a Set for Fast Lookup
    gt_cells_set = {
        (gt["start_row"], gt["start_col"], gt["end_row"], gt["end_col"], gt["text"].strip()) for gt in ground_truth_cells
    }

    # ✅ Step 4: Identify Incorrect Extractions
    incorrect_extractions = []
    incorrect_bboxes = []  # Store bounding boxes of incorrect extractions for visualization
    
    for pred in extracted_cells:
        pred_tuple = (pred["start_row"], pred["start_col"], pred["end_row"], pred["end_col"], pred["text"].strip())

        if pred_tuple not in gt_cells_set:
            incorrect_extractions.append(pred)
            incorrect_bboxes.append(pred["bbox"])  # Store bounding box

    # ✅ Step 5: Compute Metrics
    total_extracted = len(extracted_cells)  # Total number of extracted cells
    total_gt_cells = len(ground_truth_cells)  # Total number of actual table cells
    incorrect_count = len(incorrect_extractions)  # Number of incorrect extractions
    correct_count = total_extracted - incorrect_count  # Number of correct extractions

    # ✅ Compute precision, recall, and F1
    error_rate = incorrect_count / total_extracted if total_extracted > 0 else 0
    data_accuracy = correct_count / total_extracted if total_extracted > 0 else 0
   
    # ✅ Store results
    before_uq_results.append({
        "image_path": image_path,
        "data_accuracy_before_uq": data_accuracy,
        "error_rate_before_uq": error_rate
    })

    # ✅ Debugging Output
    print(f"📝 Image: {image_path}")
    print(f"🔴 Total Extracted Cells: {total_extracted}")
    print(f"✅ Total GT Cells: {total_gt_cells}")
    print(f"⚠️ Incorrect Extractions: {incorrect_count}")
    print(f"✅ Correct Extractions: {correct_count}")
    print(f"📊 Data Accuracy: {data_accuracy:.4f},  Error Rate: {error_rate}")

    # ✅ Generate visualization
    highlight_incorrect_extractions(image_path, incorrect_bboxes, viz_dir)

    return incorrect_extractions


def highlight_incorrect_extractions(image_path, incorrect_bboxes, viz_dir):
    """
    Generate and save a visualization of incorrectly extracted cells using bounding boxes.
    """
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)

    for bbox in incorrect_bboxes:
        x1, y1, x2, y2 = bbox  # Extract bounding box coordinates
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # ✅ Save visualization
    save_path = os.path.join(viz_dir, os.path.basename(image_path).replace('.png', '_errors.png'))
    plt.savefig(save_path)
    plt.close()

    print(f"📸 Incorrect Extractions Visualization Saved: {save_path}")

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

    # Initialize TSR & OCR models
    tsr_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    # Process all test images

    for image_path in test_images:
        print(f"=====================Processing {image_path}============================")
        #image_path = row['image_name']
        base_name = os.path.basename(image_path)    #(row["image_name"])
        gt_path = os.path.join(xml_dir, base_name[:-3] + "xml")  # Assuming ground truth paths are stored in the CSV
        compute_before_uq_metrics(image_path, gt_path, viz_dir, ocr_model, tsr_model)

    # Save the results
    before_uq_df = pd.DataFrame(before_uq_results)
    before_uq_df.to_csv(os.path.join(out_dir, "before_uq_" + args.domain_name + ".csv"), index=False)

    # Display summary
    print("========== Before UQ Evaluation ==========")
    print(before_uq_df.describe())