.PHONY: lint format check test clean install help

# Install all dependencies (works on both laptop and HPC)
install:
	uv sync
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "macOS detected - installing pip and vllm separately..."; \
		uv pip install --python .venv/bin/python pip && \
		.venv/bin/pip install vllm==0.12.0; \
	fi

# Run all linting checks
lint:
	uv run ruff check .
	uv run mypy .

# Auto-fix linting issues where possible
format:
	uv run ruff check --fix .
	uv run ruff format .

# Run comprehensive checks (lint + tests)
check: lint test

# Run tests
test:
	uv run pytest

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv uv.lock

# Show help
help:
	@echo "Available targets:"
	@echo "  install       - Install all dependencies (works on both laptop and HPC)"
	@echo "  lint          - Run linting checks (ruff + mypy)"
	@echo "  format        - Auto-fix linting issues and format code"
	@echo "  check         - Run lint and tests"
	@echo "  test          - Run tests"
	@echo "  clean         - Clean cache files and uv artifacts"
	@echo "  help          - Show this help message"
