.PHONY: clean format check install 


clean:
	rm -rf __pycache__

format:
	black *.py
	blackdoc *.py
	isort *.py

check:
	black *.py --check --diff
	blackdoc *.py --check
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 *.py
	mypy --config pyproject.toml *.py
	isort *.py --check --diff

install:
	git submodule update --init
	poetry install
