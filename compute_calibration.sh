#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA} Computing Calibration Data${NC}"
    echo -e "${MAGENTA}========================================${NC}"
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

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to wait for user input
wait_for_user() {
    echo
    read -p "Press Enter to continue..."
}

# Function to check if container exists
container_exists() {
    docker ps -a --format "table {{.Names}}" | grep -q "table_extraction_uq_app"
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "table_extraction_uq_app"
}

# Start calibration process
print_header

# Check if container exists, if not start it
if ! container_exists; then
    print_info "Starting Docker container..."
    docker-compose up -d
    sleep 10
else
    # Check if container is running
    if ! container_running; then
        print_info "Starting Docker container..."
        docker-compose start
        sleep 5
    fi
fi

# Run validation first
print_info "Running setup validation..."
docker exec table_extraction_uq_app python3 /app/src/validate_setup.py

echo
# Ask for user confirmation
while true; do
    read -p "Continue with calibration computation? (y/n): " continue_choice
    case $continue_choice in
        [Yy]* ) 
            break
            ;;
        [Nn]* ) 
            echo "Calibration computation cancelled."
            wait_for_user
            exit 0
            ;;
        * ) 
            echo "Please answer yes (y) or no (n)."
            ;;
    esac
done

echo
print_info "Computing APS conformal calibration data..."
print_warning "This may take 10-30 minutes depending on the number of images..."
echo

# Run calibration computation
docker exec table_extraction_uq_app python3 /app/src/compute_calibration_data.py

# Check if calibration computation was successful
if [ $? -eq 0 ]; then
    echo
    print_success "Calibration computation completed!"
    echo
    echo "Generated files:"
    echo "  - data/calibration_data/calibration_scores_aps.npy"
    echo "  - data/calibration_data/calibration_metadata.json"
    echo
    echo "You can now run the Streamlit app: ./run_app.sh"
else
    print_error "Calibration computation failed!"
    echo "Check the error messages above."
fi

echo
wait_for_user