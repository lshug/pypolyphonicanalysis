.DEFAULT_GOAL:=all
black = black .
mypy = mypy --config-file pyproject.toml --no-incremental .
ruff = ruff check .

.PHONY: format
format:
	$(black)
	$(ruff) --exit-zero --fix-only
	@echo "Done formatting"
	$(ruff)
	$(mypy)
