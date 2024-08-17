# --------------------------------------------------------------- venv

.PHONY: venv-install # install venv environment
venv-install:
	python -m venv venv;
	@bash -c '\
		source venv/bin/activate; \
		pip install --upgrade pip; \
		pip install -r requirements.txt; \
	'
	@echo "\n\n# To activate this environment, use\n#\n#     $ source venv/bin/activate\n#\n# To deactivate an active environment, use\n#\n#     $ deactivate"

.PHONY: venv-clean # remove venv environment
venv-clean:
	rm -rf venv

# --------------------------------------------------------------- conda

.PHONY: conda-get-yaml # generate an environment yaml file
conda-get-yaml:
	conda update -n base -c defaults conda
	# conda config --env --set subdir osx-64
	# conda config --env --set subdir osx-arm64
	conda config --set auto_activate_base false
	conda info
	@bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda create --yes --name con python=3.11; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate con; \
		\
		pip install -r requirements.txt; \
		\
		conda env export --no-builds | grep -v "prefix:" > conda-environment.yml; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
		conda remove --yes --name con --all; \
	'

.PHONY: conda-install # install conda environment from yaml file
conda-install:
	@bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda env create --file conda-environment.yml; \
	'

.PHONY: conda-clean # remove conda environment
conda-clean:
	# conda clean --all
	@bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda remove --yes --name con --all; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
	'

# --------------------------------------------------------------- utils

.PHONY: fmt # format and remove unused imports
fmt:
	# pip install isort
	# pip install ruff
	# pip install autoflake

	isort .
	autoflake --remove-all-unused-imports --recursive --in-place .
	ruff format --config line-length=500 .

.PHONY: sec # check for common vulnerabilities
sec:
	pip install bandit
	pip install safety
	
	bandit -r .
	safety check --full-report

.PHONY: reqs # generate requirements.txt file
reqs:
	pip install pipreqs
	rm -rf requirements.txt
	pipreqs . --mode no-pin

.PHONY: up # pull remote changes and push local changes
up:
	git pull
	git add .
	if [ -z "$(msg)" ]; then git commit -m "up"; else git commit -m "$(msg)"; fi
	git push

.PHONY: help # generate help message
help:
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
