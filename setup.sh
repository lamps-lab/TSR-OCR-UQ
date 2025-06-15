#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} Table Extraction and UQ Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
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

# Function to wait for user input
wait_for_user() {
    echo
    read -p "Press Enter to continue..."
}

# Start setup
print_header

# Check if Docker is running
echo "Checking Docker..."
if ! docker version >/dev/null 2>&1; then
    print_error "Docker is not running or not installed!"
    echo
    echo "Please:"
    echo "1. Install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    echo "2. Start Docker Desktop"
    echo "3. Run this script again"
    echo
    wait_for_user
    exit 1
fi
print_success "Docker is running"

# Create data directories
echo
echo "Creating data directories..."
mkdir -p data
mkdir -p data/input_images
mkdir -p data/calibration_data
mkdir -p models_cache

# Create domain directories
for domain in MatSci Biology CompSci ICDAR; do
    mkdir -p "data/input_images/$domain"
    mkdir -p "data/input_images/$domain/images"
done

print_success "Directories created"

# Check for required files
echo
echo "Checking required files..."

if [ ! -f "Dockerfile" ]; then
    print_error "Missing Dockerfile"
    wait_for_user
    exit 1
fi

if [ ! -f "docker-compose.yml" ]; then
    print_error "Missing docker-compose.yml"
    wait_for_user
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    print_error "Missing requirements.txt"
    wait_for_user
    exit 1
fi

if [ ! -f "src/tsr_ocr.py" ]; then
    print_error "Missing src/tsr_ocr.py"
    echo "Please add your table extraction script to src/ folder"
    wait_for_user
    exit 1
fi

print_success "Required files found"

# Build Docker image
echo
echo "Building Docker image (this may take 5-10 minutes)..."
docker-compose build

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully!"
else
    print_error "Docker build failed!"
    wait_for_user
    exit 1
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} 🎉 Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "Next steps:"
echo "1. Add your images to data/input_images/[domain]/images/ folders"
echo "2. Add domains_test_images.json to data/ folder"
echo "3. Run: ./validate.sh (optional)"
echo "4. Run: ./compute_calibration.sh"
echo "5. Run: ./run_app.sh"
echo
wait_for_user