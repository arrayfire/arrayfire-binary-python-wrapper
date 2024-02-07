SRC = arrayfire_wrapper

ifeq ($(shell uname),Darwin)
ifeq ($(shell which gsed),)
$(error Please install GNU sed with 'brew install gnu-sed')
else
SED = gsed
endif
else
SED = sed
endif

.PHONY : version
version : 
	@python -c 'from arrayfire_wrapper.version import VERSION; print(f"ArrayFire Python Wrapper v{VERSION}")'

.PHONY : install
install :
	pip install --upgrade pip
	pip install pip-tools
	pip-compile requirements.txt -o final_requirements.txt --allow-unsafe --rebuild --verbose
	pip install -r final_requirements.txt

.PHONY : build
build :
	python -m build

# Testing

.PHONY : code-style
code-style :
	black --check arrayfire_wrapper tests

.PHONY : linting
linting :
	flake8 --count --show-source --statistics arrayfire_wrapper tests

.PHONY : import-check
import-check :
	isort --check arrayfire_wrapper tests

.PHONY : typecheck
typecheck :
	mypy arrayfire_wrapper tests --cache-dir=/dev/null

.PHONY : tests
tests :
	pytest --color=yes -v -rf --durations=40 \
			--cov-config=.coveragerc \
			--cov=$(SRC) \
			--cov-report=html

# Cleaning

.PHONY : clean
clean :
	rm -rf .pytest_cache/
	rm -rf arrayfire_wrapper.egg-info/
	rm -rf dist/
	rm -rf build/
	rm final_requirements.txt
	find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
