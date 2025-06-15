#!/usr/bin/env python3
"""
Script to compute APS conformal calibration data from all images excluding test images.

This script:
1. Scans all domain folders (MatSci, Biology, CompSci, ICDAR) for images
2. Removes test images specified in the JSON file
3. Computes calibration data using extract_tsr_ocr_confidences
4. Calculates APS conformal scores for use in Streamlit app
5. Saves calibration scores as numpy file for fast loading
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import AutoModelForObjectDetection
from paddleocr import PaddleOCR
from tsr_ocr import extract_tsr_ocr_confidences
from utils import compute_calibration_scores
from score_functions import aps_conformal_score


def get_args():
    parser = argparse.ArgumentParser(description="Compute APS conformal calibration data")
    parser.add_argument("--input_dir", default="/app/data/input_images", 
                        help="Path to directory containing domain folders")
    parser.add_argument("--test_images_json", default="/app/data/domains_with_thresholds.json", 
                        help="JSON file containing test image names to exclude")
    parser.add_argument("--output_dir", default="/app/data/calibration_data", 
                        help="Directory to save calibration data")
    parser.add_argument("--domains", nargs='+', default=["MatSci", "Biology", "CompSci", "ICDAR"],
                        help="Domain folders to process")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of calibration images to use")
    args = parser.parse_args()
    return args


def get_supported_extensions():
    """Get list of supported image file extensions."""
    return {'.jpg', '.jpeg', '.png'}


def load_test_images(json_path):
    """Load test image names from JSON file."""
    print(f"Loading test images from: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"Warning: Test images JSON not found: {json_path}")
        return set()
    
    try:
        with open(json_path, 'r') as f:
            domains_data = json.load(f)
        
        test_images = set()
        for domain, data in domains_data.items():
            if 'test_data' in data:
                for item in data['test_data']:
                    test_images.add(item['image_name'])
            elif 'test_images' in data:
                test_images.update(data['test_images'])
        
        print(f"Found {len(test_images)} test images to exclude")
        return test_images
        
    except Exception as e:
        print(f"Error loading test images JSON: {e}")
        return set()


def collect_calibration_images(input_dir, domains, test_images):
    """Collect all calibration images (excluding test images)."""
    print(f"Collecting calibration images from: {input_dir}")
    
    calibration_paths = []
    supported_extensions = get_supported_extensions()
    
    for domain in domains:
        domain_images_dir = Path(input_dir) / domain / "images"
        
        if not domain_images_dir.exists():
            print(f"Warning: Domain directory not found: {domain_images_dir}")
            continue
        
        # Find all image files
        image_files = [
            f for f in domain_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        # Filter out test images
        domain_calibration = []
        for img_file in image_files:
            if img_file.name not in test_images:
                domain_calibration.append(str(img_file))
        
        print(f"  {domain}: {len(domain_calibration)} calibration images "
              f"(from {len(image_files)} total, excluded {len(image_files) - len(domain_calibration)} test images)")
        
        calibration_paths.extend(domain_calibration)
    
    print(f"Total calibration images: {len(calibration_paths)}")
    return calibration_paths


def main():
    args = get_args()
    
    print("=" * 60)
    print("APS CONFORMAL CALIBRATION DATA COMPUTATION")
    print("=" * 60)
    
    # Load models
    print("Loading models...")
    try:
        tsr_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-all"
        )
        ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Load test images to exclude
    test_images = load_test_images(args.test_images_json)
    
    # Collect calibration images
    calibration_paths = collect_calibration_images(args.input_dir, args.domains, test_images)
    
    if not calibration_paths:
        print("❌ No calibration images found!")
        return
    
    # Limit images if specified
    if args.max_images and len(calibration_paths) > args.max_images:
        print(f"Limiting to {args.max_images} images (from {len(calibration_paths)} available)")
        calibration_paths = calibration_paths[:args.max_images]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract features from calibration images
    print(f"Extracting features from {len(calibration_paths)} images...")
    calibration_data = []
    failed_count = 0
    
    for image_path in tqdm(calibration_paths, desc="Processing images"):
        try:
            extracted_cells = extract_tsr_ocr_confidences(image_path, tsr_model, ocr_model)
            if extracted_cells:
                calibration_data.append(extracted_cells)
            else:
                failed_count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            failed_count += 1
    
    print(f"Successfully processed: {len(calibration_data)} images")
    print(f"Failed extractions: {failed_count} images")
    
    if not calibration_data:
        print("❌ No calibration data extracted!")
        return
    
    # Compute APS conformal scores
    print("Computing APS conformal scores...")
    try:
        cal_scores = np.array(compute_calibration_scores(calibration_data, aps_conformal_score))
        
        # Save calibration scores as numpy file
        scores_path = os.path.join(args.output_dir, "calibration_scores_aps.npy")
        np.save(scores_path, cal_scores)
        
        print(f"✅ Saved calibration scores: {scores_path}")
        print(f"   Shape: {cal_scores.shape}")
        print(f"   Range: {cal_scores.min():.4f} - {cal_scores.max():.4f}")
        print(f"   Mean: {cal_scores.mean():.4f}")
        print(f"   Std: {cal_scores.std():.4f}")
        
    except Exception as e:
        print(f"❌ Error computing calibration scores: {e}")
        return
    
    # Save metadata
    metadata = {
        "total_calibration_images": len(calibration_paths),
        "successful_extractions": len(calibration_data),
        "failed_extractions": failed_count,
        "total_cells_extracted": sum(len(cells) for cells in calibration_data),
        "domains_used": args.domains,
        "score_function": "aps_conformal",
        "score_statistics": {
            "count": len(cal_scores),
            "min": float(cal_scores.min()),
            "max": float(cal_scores.max()),
            "mean": float(cal_scores.mean()),
            "std": float(cal_scores.std())
        }
    }
    
    metadata_path = os.path.join(args.output_dir, "calibration_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ CALIBRATION DATA COMPUTATION COMPLETED")
    print("=" * 60)
    print(f"Files created in {args.output_dir}:")
    print("  - calibration_scores_aps.npy")
    print("  - calibration_metadata.json")
    print("\nYour Streamlit app can now use real calibration data!")


if __name__ == "__main__":
    main()