# --------------------------------------------------------------- venv

.PHONY: init # init .venv with some reqs to get started
init:
	# init venv
	pip install uv
	rm -rf .venv
	uv venv

	# get reqs
	rm -rf requirements.txt requirements.in
	pipreqs . --mode no-pin --encoding latin-1
	mv requirements.txt requirements.in

	# install reqs
	uv pip compile requirements.in -o requirements.txt
	uv pip install -r requirements.txt

	# cleanup
	rm -rf requirements.txt requirements.in

.PHONY: lock # freeze pip and lock reqs
lock:
	uv pip freeze | uv pip compile - -o requirements.txt

# --------------------------------------------------------------- utils

.PHONY: fmt # format codebase
fmt:
	uv pip install isort
	uv pip install ruff
	uv pip install autoflake

	isort .
	autoflake --remove-all-unused-imports --recursive --in-place .
	ruff format --config line-length=500 .

.PHONY: sec # check for vulns
sec:
	uv pip install bandit
	uv pip install safety
	
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
