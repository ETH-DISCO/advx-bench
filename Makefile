# --------------------------------------------------------------- venv

.PHONY: venv-install # install venv environment
venv-install:
	pip install --upgrade pip
	python -m venv venv;
	@bash -c '\
		source venv/bin/activate; \
		pip install -r requirements.txt; \
	'
	@echo "\n\n# To activate this environment, use\n#\n#     $ source venv/bin/activate\n#\n# To deactivate an active environment, use\n#\n#     $ deactivate"

.PHONY: venv-clean # remove venv environment
venv-clean:
	rm -rf venv

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
