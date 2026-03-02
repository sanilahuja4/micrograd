# Create virtual environment and install dependencies
venv:
	uv venv

# Activate (not strictly needed via make, but reminder)
activate:
	@echo "Run: source .venv/bin/activate"


test:
	uv run pytest tests/

# ---------------------------
# Build & Packaging
# ---------------------------

build:
	uv python -m build

clean:
	rm -rf dist build *.egg-info

rebuild: clean build