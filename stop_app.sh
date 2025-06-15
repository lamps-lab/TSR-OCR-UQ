#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${ORANGE}========================================${NC}"
    echo -e "${ORANGE} Stopping Application${NC}"
    echo -e "${ORANGE}========================================${NC}"
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

# Start shutdown process
print_header

print_info "Stopping Docker containers..."
docker-compose down

# Check if the operation was successful
if [ $? -eq 0 ]; then
    print_success "Application stopped successfully!"
else
    print_error "Error stopping application!"
fi

# Wait for user input before closing
wait_for_user