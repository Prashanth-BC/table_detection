#!/bin/bash
set -e

# YOLO11n Table Detection - Complete Training Pipeline
# Automatically sets up, downloads data, and runs/resumes training

echo "🚀 YOLO11n Table Detection Training Pipeline"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python packages
check_python_package() {
    python -c "import $1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    if ! command_exists python; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi

    # Check Python version
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"

    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found. Please run from project root."
        exit 1
    fi

    print_status "Prerequisites check completed"
}

# Install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."

    # Install requirements
    pip install -r requirements.txt --quiet

    print_status "Dependencies installed"
}

# Setup platform-specific directories
setup_directories() {
    print_info "Setting up platform-specific directories..."

    python scripts/platform_detector.py

    print_status "Directories setup completed"
}

# Check for existing data
check_existing_data() {
    if [ -f "./data/yolo/dataset.yaml" ] && [ -d "./data/yolo/train" ] && [ -d "./data/yolo/val" ]; then
        print_status "Existing dataset found"
        return 0
    else
        print_warning "No existing dataset found"
        return 1
    fi
}

# Prepare training data
prepare_data() {
    print_info "Preparing training data..."

    if check_existing_data; then
        echo "Dataset already exists. Options:"
        echo "1) Use existing data"
        echo "2) Regenerate sample data"
        echo "3) Download real datasets"

        # Auto-select option 1 if running in non-interactive mode
        if [ -t 0 ]; then
            read -p "Choose option (1-3) [default: 1]: " data_option
        else
            data_option=1
            print_info "Non-interactive mode: using existing data (option 1)"
        fi

        case ${data_option:-1} in
            1)
                print_info "Using existing dataset"
                return 0
                ;;
            2)
                print_info "Regenerating sample dataset..."
                rm -rf ./data/raw/* ./data/yolo/* 2>/dev/null || true
                ;;
            3)
                print_info "Downloading real datasets..."
                python scripts/dataset_downloader.py
                ;;
        esac
    fi

    # Create sample dataset if no data exists
    if ! check_existing_data; then
        print_info "Creating sample dataset (50 images)..."
        python create_sample_dataset.py

        print_info "Converting data to YOLO format..."
        python scripts/data_converter.py
    fi

    print_status "Data preparation completed"
}

# Check for existing checkpoints
check_checkpoints() {
    best_checkpoint="./checkpoints/best.pt"
    last_checkpoint="./checkpoints/last.pt"

    if [ -f "$best_checkpoint" ]; then
        print_status "Found best checkpoint: $best_checkpoint"
        echo "Best checkpoint size: $(du -h "$best_checkpoint" | cut -f1)"
        return 0
    elif [ -f "$last_checkpoint" ]; then
        print_status "Found last checkpoint: $last_checkpoint"
        echo "Last checkpoint size: $(du -h "$last_checkpoint" | cut -f1)"
        return 0
    else
        print_info "No existing checkpoints found - will train from scratch"
        return 1
    fi
}

# Run training
run_training() {
    print_info "Starting YOLO11n training..."

    # Check for existing checkpoints and ask user preference
    if check_checkpoints; then
        echo ""
        echo "Training options:"
        echo "1) Resume from existing checkpoint (recommended)"
        echo "2) Start fresh training (overwrites existing)"

        # Auto-select option 1 if running in non-interactive mode
        if [ -t 0 ]; then
            read -p "Choose option (1-2) [default: 1]: " training_option
        else
            training_option=1
            print_info "Non-interactive mode: resuming from checkpoint (option 1)"
        fi

        case ${training_option:-1} in
            1)
                print_info "Resuming training from checkpoint..."
                python train_table_detection.py --resume
                ;;
            2)
                print_warning "Starting fresh training (existing checkpoints will be overwritten)"
                python train_table_detection.py --resume=False
                ;;
        esac
    else
        print_info "Starting fresh training..."
        python train_table_detection.py
    fi
}

# Display results
show_results() {
    print_info "Training completed! Checking results..."

    if [ -f "./checkpoints/best.pt" ]; then
        print_status "Best model saved: ./checkpoints/best.pt"
        echo "Model size: $(du -h ./checkpoints/best.pt | cut -f1)"
    fi

    if [ -f "progress.json" ]; then
        print_info "Extracting training metrics..."
        python -c "
import json
try:
    with open('progress.json', 'r') as f:
        progress = json.load(f)
    if 'evaluation' in progress and 'results' in progress['evaluation']:
        results = progress['evaluation']['results']
        print(f'📊 Final Results:')
        print(f'   mAP50: {results.get(\"map50\", 0):.3f}')
        print(f'   mAP:   {results.get(\"map\", 0):.3f}')
        print(f'   Precision: {results.get(\"precision\", 0):.3f}')
        print(f'   Recall: {results.get(\"recall\", 0):.3f}')
    else:
        print('Results not available in progress.json')
except Exception as e:
    print(f'Could not read results: {e}')
"
    fi

    echo ""
    print_status "Training pipeline completed successfully!"
    echo ""
    echo "🎯 Next steps:"
    echo "   • Model ready for inference: ./checkpoints/best.pt"
    echo "   • View training results: ./results/table_detection/"
    echo "   • Resume training anytime: ./run_training.sh"
    echo ""
    echo "📖 Usage example:"
    echo "   from ultralytics import YOLO"
    echo "   model = YOLO('./checkpoints/best.pt')"
    echo "   results = model('table_image.jpg')"
}

# Error handling
handle_error() {
    print_error "Training pipeline failed!"
    echo "Check the error messages above for details."
    echo "Common solutions:"
    echo "  • Ensure you have sufficient disk space"
    echo "  • Check GPU memory if using CUDA"
    echo "  • Verify Python dependencies are installed"
    exit 1
}

# Set up error handling
trap 'handle_error' ERR

# Main execution
main() {
    echo "Starting at: $(date)"
    echo ""

    # Run pipeline steps
    check_prerequisites
    install_dependencies
    setup_directories
    prepare_data
    run_training
    show_results

    echo "Completed at: $(date)"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "YOLO11n Table Detection Training Pipeline"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --data-only         Only prepare data, don't train"
        echo "  --train-only        Only run training (skip data preparation)"
        echo "  --fresh             Force fresh training (ignore checkpoints)"
        echo ""
        echo "Examples:"
        echo "  $0                  # Full pipeline"
        echo "  $0 --data-only      # Just prepare data"
        echo "  $0 --train-only     # Just run training"
        echo "  $0 --fresh          # Fresh training from scratch"
        exit 0
        ;;
    --data-only)
        print_info "Data preparation mode"
        check_prerequisites
        install_dependencies
        setup_directories
        prepare_data
        print_status "Data preparation completed"
        exit 0
        ;;
    --train-only)
        print_info "Training only mode"
        check_prerequisites
        run_training
        show_results
        exit 0
        ;;
    --fresh)
        print_info "Fresh training mode (ignoring existing checkpoints)"
        check_prerequisites
        install_dependencies
        setup_directories
        prepare_data
        print_info "Starting fresh training..."
        python train_table_detection.py --resume=False
        show_results
        exit 0
        ;;
    "")
        # No arguments - run full pipeline
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac