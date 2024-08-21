# --------------------------------------------------------------- venv

# check out: https://github.com/astral-sh/uv

.PHONY: init # initialize venv
init:
	# get requirements.in
	pip install pip --upgrade
	pip install pipreqs
	rm -rf requirements.txt requirements.in
	pipreqs . --mode no-pin --encoding latin-1
	mv requirements.txt requirements.in

	# get requirements.txt
	pip install pip-tools
	pip-compile requirements.in -o requirements.txt -vvv
	
	# install everything in venv
	rm -rf .venv
	python3 -m venv .venv
	@bash -c "source .venv/bin/activate && pip install -r requirements.txt"

.PHONY: lock # freeze pip and lock reqs
lock:
	@bash -c "source .venv/bin/activate && pip freeze > requirements.in"
	pip-compile requirements.in -o requirements.txt

# --------------------------------------------------------------- utils

.PHONY: fmt # format codebase
fmt:
	# pip install isort
	# pip install ruff
	# pip install autoflake

	isort .
	autoflake --remove-all-unused-imports --recursive --in-place .
	ruff format --config line-length=500 .

.PHONY: sec # check for vulns
sec:
	pip install bandit
	pip install safety
	
	bandit -r .
	safety check --full-report

.PHONY: up # pull and push changes
up:
	git pull
	git add .
	if [ -z "$(msg)" ]; then git commit -m "up"; else git commit -m "$(msg)"; fi
	git push

.PHONY: help # generate help message
help:
	@echo "Usage: make [target]\n"
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
