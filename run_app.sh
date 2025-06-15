#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BRIGHT_RED='\033[1;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BRIGHT_RED}========================================${NC}"
    echo -e "${BRIGHT_RED} Starting Streamlit App${NC}"
    echo -e "${BRIGHT_RED}========================================${NC}"
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

print_url() {
    echo -e "${CYAN}🌐 $1${NC}"
}

# Function to wait for user input
wait_for_user() {
    echo
    read -p "Press Enter to continue..."
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "table_extraction_uq_app"
}

# Start app
print_header

# Check if calibration data exists
if [ ! -f "data/calibration_data/calibration_scores_aps.npy" ]; then
    print_error "Calibration data not found!"
    echo "Please run: ./compute_calibration.sh first"
    wait_for_user
    exit 1
fi

# Start container if not running
if ! container_running; then
    print_info "Starting Docker container..."
    docker-compose up -d
    sleep 10
fi

print_info "Starting Streamlit app..."
echo
print_url "Opening app at: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the app"
echo

# Function to handle cleanup on script exit
cleanup() {
    echo
    print_info "App stopped by user"
    exit 0
}

# Set up signal handling for graceful shutdown
trap cleanup SIGINT SIGTERM

# Run Streamlit app
docker exec table_extraction_uq_app streamlit run /app/src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0