.PHONY: help install checks tests clean

UV_VERSION := 0.8.23

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install uv (v$(UV_VERSION)) and all dependencies
	@NEEDS_INSTALL=false; \
	if command -v uv >/dev/null 2>&1; then \
		CURRENT_VERSION=$$(uv --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'); \
		if [ "$$CURRENT_VERSION" != "$(UV_VERSION)" ]; then \
			echo "⚠️  uv $$CURRENT_VERSION installed, but $(UV_VERSION) required"; \
			NEEDS_INSTALL=true; \
		else \
			echo "✅ uv $(UV_VERSION) is already installed"; \
		fi; \
	else \
		echo "📦 uv not found"; \
		NEEDS_INSTALL=true; \
	fi; \
	if [ "$$NEEDS_INSTALL" = "true" ]; then \
		echo "📦 Installing uv $(UV_VERSION)..."; \
		curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		if [ -f "$$HOME/.bashrc" ]; then echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> "$$HOME/.bashrc"; fi; \
		if [ -f "$$HOME/.zshrc" ]; then echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> "$$HOME/.zshrc"; fi; \
		echo "✅ uv $(UV_VERSION) installed"; \
		echo "⚠️  Restart your shell or run: export PATH=\"$$HOME/.local/bin:\$$PATH\""; \
	fi
	@echo "📦 Installing dependencies..."
	@uv sync --all-groups --all-extras --frozen
	@echo "🔗 Installing pre-commit hooks..."
	@PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL="" uv run pre-commit install --install-hooks
	@echo "✅ Installation complete!"

checks: ## Run pre-commit checks on all files
	@echo "🔍 Running checks..."
	@PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL="" uv run pre-commit run --all-files

tests: ## Run tests with coverage
	@echo "🧪 Running tests..."
	@uv run pytest src/tests/ -vv

clean: ## Clean up temporary files and caches
	@echo "🧹 Cleaning up..."
	@PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL="" uv run pre-commit clean 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build *.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"
