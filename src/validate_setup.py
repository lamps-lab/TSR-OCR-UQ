#!/usr/bin/env python3
"""
Validation script to check setup before calibration computation.
"""

import os
import json
from pathlib import Path
from collections import defaultdict


def get_supported_extensions():
    return {'.jpg', '.jpeg', '.png'}


def validate_folder_structure():
    """Check if required folders exist."""
    print("🔍 Validating folder structure...")
    
    required_folders = [
        "/app/data/input_images",
        "/app/data/input_images/MatSci/images",
        "/app/data/input_images/Biology/images", 
        "/app/data/input_images/CompSci/images",
        "/app/data/input_images/ICDAR/images",
    ]
    
    missing_folders = []
    for folder in required_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print("❌ Missing folders:")
        for folder in missing_folders:
            print(f"   {folder}")
        return False
    else:
        print("✅ All required folders found")
        return True


def count_images():
    """Count images in each domain."""
    print("\n📊 Counting images in each domain...")
    
    domains = ["MatSci", "Biology", "CompSci", "ICDAR"]
    supported_ext = get_supported_extensions()
    domain_counts = {}
    
    total_images = 0
    for domain in domains:
        images_dir = Path(f"/app/data/input_images/{domain}/images")
        
        if images_dir.exists():
            image_files = [
                f for f in images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in supported_ext
            ]
            count = len(image_files)
            domain_counts[domain] = count
            total_images += count
            print(f"   {domain:8s}: {count:4d} images")
        else:
            domain_counts[domain] = 0
            print(f"   {domain:8s}: ❌ Folder not found")
    
    print(f"   {'Total':8s}: {total_images:4d} images")
    return domain_counts, total_images


def load_test_images():
    """Load and validate test images JSON."""
    print("\n🧪 Validating test images configuration...")
    
    json_path = "/app/data/domains_with_thresholds.json"
    
    if not os.path.exists(json_path):
        print("❌ domains_with_thresholds.json not found!")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        test_images = {}
        total_test = 0
        
        for domain, domain_data in data.items():
            if 'test_data' in domain_data:
                test_list = [item['image_name'] for item in domain_data['test_data']]
            elif 'test_images' in domain_data:
                test_list = domain_data['test_images']
            else:
                test_list = []
            
            test_images[domain] = test_list
            total_test += len(test_list)
            print(f"   {domain:8s}: {len(test_list):4d} test images")
        
        print(f"   {'Total':8s}: {total_test:4d} test images")
        print("✅ Test images JSON loaded successfully")
        return test_images
        
    except Exception as e:
        print(f"❌ Error loading test images JSON: {e}")
        return {}


def calculate_calibration_available(domain_counts, test_images):
    """Calculate available calibration images."""
    print("\n📈 Calculating calibration data availability...")
    
    total_calibration = 0
    issues = []
    
    for domain in ["MatSci", "Biology", "CompSci", "ICDAR"]:
        total_domain = domain_counts.get(domain, 0)
        test_domain = len(test_images.get(domain, []))
        calibration_domain = max(0, total_domain - test_domain)
        total_calibration += calibration_domain
        
        print(f"   {domain:8s}: {calibration_domain:4d} calibration images "
              f"({total_domain} total - {test_domain} test)")
        
        if total_domain > 0 and calibration_domain < 5:
            issues.append(f"{domain} has only {calibration_domain} calibration images")
        elif total_domain == 0:
            issues.append(f"{domain} has no images")
    
    print(f"   {'Total':8s}: {total_calibration:4d} calibration images")
    
    return total_calibration, issues


def check_required_files():
    """Check if required Python files exist."""
    print("\n📄 Checking required Python files...")
    
    required_files = [
        "/app/src/tsr_ocr.py",
        "/app/src/utils.py", 
        "/app/src/score_functions.py",
        "/app/src/compute_calibration_data.py",
        "/app/src/streamlit_app.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {os.path.basename(file_path)}")
        else:
            print(f"   ❌ {os.path.basename(file_path)} missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0


def provide_recommendations(total_calibration, issues):
    """Provide setup recommendations."""
    print("\n💡 Recommendations:")
    print("-" * 40)
    
    if total_calibration >= 200:
        print("   ✅ Excellent: Sufficient calibration data")
        print("   📝 Proceed with full calibration computation")
        return "ready"
    elif total_calibration >= 50:
        print("   ⚠️  Adequate: Limited but usable calibration data")
        print("   📝 Consider adding more training images if possible")
        return "limited"
    else:
        print("   ❌ Insufficient: Too few calibration images")
        print("   📝 Add more training images before proceeding")
        return "insufficient"


def main():
    print("=" * 60)
    print("TABLE EXTRACTION & UQ SETUP VALIDATION")
    print("=" * 60)
    
    # Validate folder structure
    if not validate_folder_structure():
        print("\n❌ Setup validation failed: Missing folders")
        return
    
    # Count images
    domain_counts, total_images = count_images()
    
    if total_images == 0:
        print("\n❌ Setup validation failed: No images found")
        print("Please add images to the domain folders")
        return
    
    # Load test images
    test_images = load_test_images()
    
    if not test_images:
        print("\n❌ Setup validation failed: No test images configuration")
        print("Please add domains_with_thresholds.json to data/ folder")
        return
    
    # Calculate calibration availability
    total_calibration, issues = calculate_calibration_available(domain_counts, test_images)
    
    # Check required files
    files_ok = check_required_files()
    
    if not files_ok:
        print("\n❌ Setup validation failed: Missing required Python files")
        return
    
    # Show issues if any
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Provide recommendations
    status = provide_recommendations(total_calibration, issues)
    
    # Final summary
    print(f"\n📋 Setup Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Calibration images: {total_calibration}")
    print(f"   Required files: {'✅ All present' if files_ok else '❌ Missing files'}")
    
    if status == "ready":
        print(f"   Status: ✅ Ready for calibration computation")
        print(f"\n🚀 Next step: Run calibration computation")
    elif status == "limited":
        print(f"   Status: ⚠️  Limited but usable")
        print(f"\n🚀 Next step: Run calibration computation (may take longer)")
    else:
        print(f"   Status: ❌ Not ready")
        print(f"\n📝 Action needed: Add more training images")


if __name__ == "__main__":
    main()