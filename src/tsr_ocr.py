import os
import torch
import json
import numpy as np
from shapely.geometry import box as shapely_box
from paddleocr import PaddleOCR
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import argparse
from pathlib import Path
from typing import Union, List, Dict, Any


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="Path to the image directory")
    parser.add_argument("--image_path", help="Path to a single table image")
    parser.add_argument("--out_dir", help="path to save the computed metrics")
    args = parser.parse_args()
    return args

# Utility Functions
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def get_table_structure(image, model, image_processor):
    # Prepare inputs for the model
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[image.size])[0]
    
    return results

def apply_table_transformer(image, str_model):
    try:
        image_processor =  AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all") # AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection") 
        results = get_table_structure(image, str_model, image_processor)
    except:
        image_processor =  AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        results = get_table_structure(image, str_model, image_processor)
    return results

# Scale TSR bboxes
def scale_bounding_boxes(bboxes, scale_x, scale_y):
    scaled_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        scaled_bboxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
    return scaled_bboxes

# Extract bboxes, texts, and confidences from OCR
def extract_separate_lists(data):
    bounding_boxes, texts, confidence_scores = [], [], []
    for line in data:
        for cell in line:
            bounding_boxes.append(cell[0])
            texts.append(cell[1][0])
            confidence_scores.append(cell[1][1])
    return bounding_boxes, texts, confidence_scores

# Collapse OCR bboxes to match TSR bboxes
def collapse_bounding_boxes(bounding_boxes):
    collapsed_boxes = []
    for box in bounding_boxes:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        collapsed_boxes.append([x_min, y_min, x_max, y_max])
    return collapsed_boxes

# Get TSR confidences for rows and columns
def get_rows_and_columns_with_confidences(results):
    rows, columns, row_confidences, col_confidences = [], [], [], []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        rounded_box = [round(coord, 2) for coord in box.tolist()]
        if label == 2:  # Row
            rows.append(rounded_box)
            row_confidences.append(score)
        elif label == 1:  # Column
            columns.append(rounded_box)
            col_confidences.append(score)
    return rows, columns, row_confidences, col_confidences

def generate_grid_cells_with_tsr_confidence(sorted_rows, sorted_columns, tsr_row_confidences, tsr_col_confidences):
    grid_cells = []
    for row_idx, row in enumerate(sorted_rows):
        row_bbox, row_confidence = row[:4], tsr_row_confidences[row_idx]
        for col_idx, col in enumerate(sorted_columns):
            col_bbox, col_confidence = col[:4], tsr_col_confidences[col_idx]
            x1, y1, x2, y2 = max(row_bbox[0], col_bbox[0]), max(row_bbox[1], col_bbox[1]), min(row_bbox[2], col_bbox[2]), min(row_bbox[3], col_bbox[3])
            if x2 > x1 and y2 > y1:
                cell_bbox = [x1, y1, x2, y2]
                grid_cells.append({
                    'bbox': cell_bbox,
                    'temp_row_idx': row_idx,
                    'temp_col_idx': col_idx,
                    'row_confidence': row_confidence,
                    'col_confidence': col_confidence,
                    'ocr_confidence': 0.0,
                    'text': ''
                })
    return grid_cells


def assign_texts_and_confidences_to_cells(grid_cells, ocr_bboxes, texts, ocr_confidences):
    ocr_polygons = [shapely_box(*bbox) for bbox in ocr_bboxes]
    for cell in grid_cells:
        cell_polygon = shapely_box(*cell['bbox'])
        cell_texts, cell_confidences = [], []
        for ocr_polygon, text, confidence in zip(ocr_polygons, texts, ocr_confidences):
            intersection = cell_polygon.intersection(ocr_polygon).area
            #if cell_polygon.intersects(ocr_polygon):
            if intersection / ocr_polygon.area > 0.5:
                cell_texts.append(text)
                cell_confidences.append(confidence)
        if cell_texts:
            cell['text'] = " ".join(cell_texts).strip()
            cell['ocr_confidence'] = np.mean(cell_confidences) if cell_confidences else 0.0
        else:
            cell["ocr_confidence"] = 0.0
    return grid_cells

# Aggregate the row and column confidences of TSR
def distribute_tsr_confidences(grid_cells, tsr_row_confidences, tsr_col_confidences):
    """
    Distribute TSR row and column confidences to each cell in the grid.
    """
    for cell in grid_cells:
        row_idx = cell['temp_row_idx']
        col_idx = cell['temp_col_idx']

        # Assign TSR row and column confidences to the cell
        tsr_row_conf = tsr_row_confidences[row_idx]
        tsr_col_conf = tsr_col_confidences[col_idx]

        # Combine row and column TSR confidences (e.g., by averaging)
        cell['tsr_combined_conf'] = (tsr_row_conf + tsr_col_conf) / 2
        cell["start_row"] = row_idx
        cell["end_row"] = row_idx
        cell["start_col"] = col_idx
        cell["end_col"] = col_idx

    # Remove temporary indices
    for cell in grid_cells:
        del cell['temp_row_idx']
        del cell['temp_col_idx']
    return grid_cells


def extract_tsr_ocr_confidences(image_path, tsr_model, ocr_model):
    """
    Extract TSR and OCR confidences aligned to grid cells.
    """
    image = load_image(image_path)

    # TSR results
    tsr_results = apply_table_transformer(image, tsr_model)
    tsr_rows, tsr_columns, tsr_row_confidences, tsr_col_confidences = get_rows_and_columns_with_confidences(tsr_results)

    # OCR results
    ocr_results = ocr_model.ocr(image_path, cls=True)
    ocr_bboxes, ocr_texts, ocr_confidences = extract_separate_lists(ocr_results)
    ocr_bboxes = collapse_bounding_boxes(ocr_bboxes)

    # Scale TSR bounding boxes to match OCR
    max_x_ocr = max(bbox[2] for bbox in ocr_bboxes)
    max_y_ocr = max(bbox[3] for bbox in ocr_bboxes)
    max_x_tsr = max(max(row[2] for row in tsr_rows), max(col[2] for col in tsr_columns))
    max_y_tsr = max(max(row[3] for row in tsr_rows), max(col[3] for col in tsr_columns))
    scale_x = max_x_ocr / max_x_tsr
    scale_y = max_y_ocr / max_y_tsr
    scaled_rows = scale_bounding_boxes(tsr_rows, scale_x, scale_y)
    scaled_columns = scale_bounding_boxes(tsr_columns, scale_x, scale_y)

    # Sort and assign indices
    sorted_rows = sorted(scaled_rows, key=lambda x: x[1])
    sorted_columns = sorted(scaled_columns, key=lambda x: x[0])
    for idx, row in enumerate(sorted_rows):
        row.append(idx)
    for idx, col in enumerate(sorted_columns):
        col.append(idx)

    # Generate grid cells
    grid_cells = generate_grid_cells_with_tsr_confidence(sorted_rows, sorted_columns, tsr_row_confidences, tsr_col_confidences)

    # Distribute TSR confidences across grid cells
    grid_cells = distribute_tsr_confidences(grid_cells, tsr_row_confidences, tsr_col_confidences)

    # Assign OCR confidences to grid cells
    grid_cells = assign_texts_and_confidences_to_cells(grid_cells, ocr_bboxes, ocr_texts, ocr_confidences)
    return grid_cells


def convert_tensors_to_serializable(obj):
    """
    Convert PyTorch tensors to Python native types for JSON serialization.
    """
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_serializable(item) for item in obj]
    else:
        return obj


def save_table_extraction_results(image_path: str, grid_cells: List[Dict], output_dir: str = None) -> str:
    """
    Save table extraction results to a JSON file.
    
    Args:
        image_path (str): Path to the input image
        grid_cells (List[Dict]): Extracted table data
        output_dir (str, optional): Directory to save JSON file. If None, saves in same directory as image.
    
    Returns:
        str: Path to the saved JSON file
    """
    # Get image info
    image_path = Path(image_path)
    image_name = image_path.stem  # filename without extension
    
    # Determine output directory
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create JSON filename
    json_filename = f"{image_name}_table_extraction.json"
    json_path = output_dir / json_filename
    
    # Prepare data for JSON serialization
    serializable_data = convert_tensors_to_serializable(grid_cells)
    
    # Create comprehensive result structure
    result_data = {
        "source_image": str(image_path),
        "extraction_metadata": {
            "total_cells": len(grid_cells),
            "rows": max([cell.get('row_idx', 0) for cell in grid_cells]) + 1 if grid_cells else 0,
            "columns": max([cell.get('col_idx', 0) for cell in grid_cells]) + 1 if grid_cells else 0,
            "cells_with_text": len([cell for cell in grid_cells if cell.get('text', '').strip()])
        },
        "table_cells": serializable_data
    }
    
    # Save to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return str(json_path)


def get_supported_image_extensions():
    """Get list of supported image file extensions."""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def process_single_image(image_path: str, tsr_model, ocr_model, output_dir: str = None) -> str:
    """
    Process a single table image and save results as JSON.
    
    Args:
        image_path (str): Path to the image file
        tsr_model: Pre-loaded TSR model
        ocr_model: Pre-loaded OCR model
        output_dir (str, optional): Directory to save results
    
    Returns:
        str: Path to the saved JSON file
    """
    print(f"Processing image: {image_path}")
    
    try:
        # Extract table data
        grid_cells = extract_tsr_ocr_confidences(image_path, tsr_model, ocr_model)
        
        # Save results
        json_path = save_table_extraction_results(image_path, grid_cells, output_dir)
        print(f"Results saved to: {json_path}")
        
        return json_path
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        raise


def process_image_directory(image_dir: str, tsr_model, ocr_model, output_dir: str = None) -> List[str]:
    """
    Process all table images in a directory and save results as JSON files.
    
    Args:
        image_dir (str): Path to directory containing images
        tsr_model: Pre-loaded TSR model
        ocr_model: Pre-loaded OCR model
        output_dir (str, optional): Directory to save results
    
    Returns:
        List[str]: List of paths to saved JSON files
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # Get all image files
    supported_extensions = get_supported_image_extensions()
    image_files = [
        f for f in image_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not image_files:
        print(f"No supported image files found in {image_dir}")
        return []
    
    print(f"Found {len(image_files)} image files to process")
    
    json_paths = []
    successful_count = 0
    failed_count = 0
    
    for image_file in image_files:
        try:
            json_path = process_single_image(str(image_file), tsr_model, ocr_model, output_dir)
            json_paths.append(json_path)
            successful_count += 1
        except Exception as e:
            print(f"Failed to process {image_file}: {str(e)}")
            failed_count += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed: {failed_count} images")
    
    return json_paths


def extract_tables_to_json(input_path: str, output_dir: str = None, 
                          tsr_model=None, ocr_model=None) -> Union[str, List[str]]:
    """
    Main function to extract table data from either a single image or directory of images.
    
    Args:
        input_path (str): Path to either a single image file or directory containing images
        output_dir (str, optional): Directory to save JSON results. If None, saves in same location as input.
        tsr_model (optional): Pre-loaded TSR model. If None, loads default model.
        ocr_model (optional): Pre-loaded OCR model. If None, loads default model.
    
    Returns:
        Union[str, List[str]]: Path to JSON file (single image) or list of JSON paths (directory)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Load models if not provided
    if tsr_model is None:
        print("Loading TSR model...")
        tsr_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
    
    if ocr_model is None:
        print("Loading OCR model...")
        ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Process based on input type
    if input_path.is_file():
        # Single image
        supported_extensions = get_supported_image_extensions()
        if input_path.suffix.lower() not in supported_extensions:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        
        return process_single_image(str(input_path), tsr_model, ocr_model, output_dir)
    
    elif input_path.is_dir():
        # Directory of images
        return process_image_directory(str(input_path), tsr_model, ocr_model, output_dir)
    
    else:
        raise ValueError(f"Input path must be a file or directory: {input_path}")


if __name__ == "__main__":
    args = get_args()
    
    # Determine input path
    if args.image_path:
        input_path = args.image_path
    elif args.image_dir:
        input_path = args.image_dir
    else:
        raise ValueError("Please provide either --image_path or --image_dir")
    
    # Extract tables and save to JSON
    try:
        results = extract_tables_to_json(
            input_path=input_path,
            output_dir=args.out_dir
        )
        
        if isinstance(results, str):
            print(f"\nTable extraction complete! Results saved to: {results}")
        else:
            print(f"\nBatch processing complete! {len(results)} JSON files created.")
            
    except Exception as e:
        print(f"Error: {str(e)}")