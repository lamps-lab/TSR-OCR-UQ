#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${WHITE}========================================${NC}"
    echo -e "${WHITE} Project Status Check${NC}"
    echo -e "${WHITE}========================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

print_section() {
    echo -e "${CYAN}$1${NC}"
}

# Function to wait for user input
wait_for_user() {
    echo
    read -p "Press Enter to continue..."
}

# Function to count files in directory
count_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        find "$dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" -o -name "*.bmp" -o -name "*.tiff" \) 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "table_extraction_uq_app"
}

# Start status check
print_header

# Docker Status
print_section "Docker Status:"
if docker version >/dev/null 2>&1; then
    print_success "Docker is running"
    
    if container_running; then
        print_success "Container is running"
    else
        print_error "Container is not running"
    fi
else
    print_error "Docker is not running"
fi

echo

# Directory Structure
print_section "Directory Structure:"
if [ -d "data/input_images" ]; then
    print_success "data/input_images/ exists"
    
    # Check each domain directory and count images
    for domain in MatSci Biology CompSci ICDAR; do
        domain_path="data/input_images/$domain/images"
        if [ -d "$domain_path" ]; then
            image_count=$(count_files "$domain_path")
            echo "  $domain: $image_count images"
        else
            print_error "  $domain/images/ missing"
        fi
    done
else
    print_error "data/input_images/ missing"
fi

echo

# Required Files
print_section "Required Files:"
if [ -f "data/domains_with_thresholds.json" ]; then
    print_success "Test images JSON found"
else
    print_error "domains_with_thresholds.json missing"
fi

# Check source files
if [ -f "src/tsr_ocr.py" ]; then
    print_success "tsr_ocr.py"
else
    print_error "tsr_ocr.py missing"
fi

if [ -f "src/utils.py" ]; then
    print_success "utils.py"
else
    print_error "utils.py missing"
fi

if [ -f "src/score_functions.py" ]; then
    print_success "score_functions.py"
else
    print_error "score_functions.py missing"
fi

echo

# Calibration Data
print_section "Calibration Data:"
if [ -f "data/calibration_data/calibration_scores_aps.npy" ]; then
    print_success "APS calibration data computed"
else
    print_error "APS calibration data not computed"
    echo "   Run: ./compute_calibration.sh"
fi

# Wait for user input before closing
wait_for_user