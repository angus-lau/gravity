#!/bin/bash
# Ad Retrieval System - Initialization Script
# Usage: ./init.sh [command]
#
# Commands:
#   setup       - Install dependencies and build indexes
#   server      - Start the FastAPI server
#   test        - Run all tests
#   test-cov    - Run tests with coverage report
#   lint        - Run linter (ruff)
#   format      - Format code (ruff format)
#   benchmark   - Run latency benchmarks
#   docker      - Build and run Docker container
#   help        - Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Print colored message
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        warn "Virtual environment not found. Creating one..."
        python3 -m venv venv
        success "Virtual environment created"
    fi
}

# Activate virtual environment
activate_venv() {
    check_venv
    source venv/bin/activate
    info "Virtual environment activated"
}

# Install dependencies
cmd_setup() {
    info "Setting up project..."
    activate_venv
    
    info "Upgrading pip..."
    pip install --upgrade pip
    
    info "Installing dependencies..."
    pip install -r requirements.txt
    
    info "Installing dev dependencies..."
    pip install -r requirements-dev.txt 2>/dev/null || warn "No requirements-dev.txt found"
    
    # Create necessary directories
    mkdir -p data app/models app/retrieval tests scripts
    
    # Check if campaign data exists, if not generate it
    if [ ! -f "data/campaigns.json" ]; then
        warn "Campaign data not found. You may need to run: python scripts/generate_campaigns.py"
    fi
    
    # Check if FAISS index exists, if not build it
    if [ ! -f "data/campaigns.index" ]; then
        warn "FAISS index not found. You may need to run: python scripts/build_index.py"
    fi
    
    success "Setup complete!"
}

# Start the FastAPI server
cmd_server() {
    activate_venv
    info "Starting FastAPI server on http://localhost:8000"
    info "API docs available at http://localhost:8000/docs"
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# Start server in production mode (no reload)
cmd_server_prod() {
    activate_venv
    info "Starting FastAPI server in production mode..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
}

# Run all tests
cmd_test() {
    activate_venv
    info "Running tests..."
    pytest tests/ -v --tb=short "$@"
}

# Run tests with coverage
cmd_test_cov() {
    activate_venv
    info "Running tests with coverage..."
    pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html
    success "Coverage report generated at htmlcov/index.html"
}

# Run specific test file or pattern
cmd_test_only() {
    activate_venv
    if [ -z "$1" ]; then
        error "Usage: ./init.sh test-only <pattern>"
    fi
    info "Running tests matching: $1"
    pytest tests/ -v -k "$1"
}

# Run linter
cmd_lint() {
    activate_venv
    info "Running linter..."
    ruff check app/ tests/ scripts/ --output-format=concise
    success "Linting complete!"
}

# Format code
cmd_format() {
    activate_venv
    info "Formatting code..."
    ruff format app/ tests/ scripts/
    ruff check app/ tests/ scripts/ --fix
    success "Formatting complete!"
}

# Run type checker
cmd_typecheck() {
    activate_venv
    info "Running type checker..."
    mypy app/ --ignore-missing-imports
}

# Run latency benchmarks
cmd_benchmark() {
    activate_venv
    local url="${1:-http://localhost:8000}"
    info "Running benchmarks against: $url"
    python scripts/benchmark.py "$url"
}

# Build Docker image
cmd_docker_build() {
    info "Building Docker image..."
    docker build -t ad-retrieval:latest .
    success "Docker image built: ad-retrieval:latest"
}

# Run Docker container
cmd_docker_run() {
    info "Running Docker container..."
    docker run -p 8000:8000 --rm ad-retrieval:latest
}

# Build and run Docker
cmd_docker() {
    cmd_docker_build
    cmd_docker_run
}

# Generate synthetic campaign data
cmd_generate_data() {
    activate_venv
    info "Generating synthetic campaign data..."
    python scripts/generate_campaigns.py
    success "Campaign data generated!"
}

# Build FAISS index
cmd_build_index() {
    activate_venv
    info "Building FAISS index..."
    python scripts/build_index.py
    success "FAISS index built!"
}

# Full rebuild: generate data + build index
cmd_rebuild() {
    cmd_generate_data
    cmd_build_index
    success "Full rebuild complete!"
}

# Clean generated files
cmd_clean() {
    info "Cleaning generated files..."
    rm -rf __pycache__ .pytest_cache .coverage htmlcov .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    success "Cleaned!"
}

# Show help
cmd_help() {
    echo "Ad Retrieval System - Initialization Script"
    echo ""
    echo "Usage: ./init.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup         Install dependencies and prepare environment"
    echo "  server        Start FastAPI server (development mode with reload)"
    echo "  server-prod   Start FastAPI server (production mode)"
    echo "  test          Run all tests"
    echo "  test-cov      Run tests with coverage report"
    echo "  test-only <p> Run tests matching pattern"
    echo "  lint          Run linter (ruff)"
    echo "  format        Format code with ruff"
    echo "  typecheck     Run mypy type checker"
    echo "  benchmark     Run latency benchmarks [url]"
    echo "  docker        Build and run Docker container"
    echo "  docker-build  Build Docker image only"
    echo "  docker-run    Run Docker container only"
    echo "  generate-data Generate synthetic campaign data"
    echo "  build-index   Build FAISS index from campaign data"
    echo "  rebuild       Full rebuild (generate data + build index)"
    echo "  clean         Remove generated/cache files"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./init.sh setup           # First time setup"
    echo "  ./init.sh server          # Start development server"
    echo "  ./init.sh test            # Run all tests"
    echo "  ./init.sh test-only eligibility  # Run eligibility tests only"
    echo "  ./init.sh benchmark http://localhost:8000"
}

# Main command dispatcher
case "${1:-help}" in
    setup)        cmd_setup ;;
    server)       cmd_server ;;
    server-prod)  cmd_server_prod ;;
    test)         shift; cmd_test "$@" ;;
    test-cov)     cmd_test_cov ;;
    test-only)    shift; cmd_test_only "$@" ;;
    lint)         cmd_lint ;;
    format)       cmd_format ;;
    typecheck)    cmd_typecheck ;;
    benchmark)    shift; cmd_benchmark "$@" ;;
    docker)       cmd_docker ;;
    docker-build) cmd_docker_build ;;
    docker-run)   cmd_docker_run ;;
    generate-data) cmd_generate_data ;;
    build-index)  cmd_build_index ;;
    rebuild)      cmd_rebuild ;;
    clean)        cmd_clean ;;
    help|--help|-h) cmd_help ;;
    *)
        error "Unknown command: $1\nRun './init.sh help' for usage"
        ;;
esac
