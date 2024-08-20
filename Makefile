# --------------------------------------------------------------- venv

.PHONY: init # initialize venv
init:
	# init venv
	pip install uv
	rm -rf .venv
	uv venv

	# install reqs
	rm -rf requirements.txt requirements.in
	pipreqs . --mode no-pin --encoding latin-1
	mv requirements.txt requirements.in

	uv pip compile requirements.in -o requirements.txt
	uv pip install -r requirements.txt

	rm -rf requirements.txt requirements.in

.PHONY: lock # freeze pip and lock reqs
lock:
	uv pip freeze | uv pip compile - -o requirements.txt

# --------------------------------------------------------------- conda

.PHONY: conda-get-yaml # convert requirements.txt to env.yaml file (idempotent)
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
		conda env export --no-builds | grep -v "prefix:" > env.yml; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
		conda remove --yes --name con --all; \
	'

.PHONY: conda-install # install conda from env.yaml file
conda-install:
	@bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda env create --file env.yml; \
	'

.PHONY: conda-clean # wipe conda environment
conda-clean:
	# conda clean --all
	@bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda remove --yes --name con --all; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
	'

# --------------------------------------------------------------- docker

.PHONY: docker-install # run docker container
docker-install:
	@echo "to exec into docker container, run: 'docker exec -it main /bin/bash'"
	docker-compose up --detach

.PHONY: docker-clean # wipe everything in docker
docker-clean:
	docker-compose down

	-docker stop $$(docker ps -a -q)
	-docker rm $$(docker ps -a -q)
	-docker rmi $$(docker images -q)
	yes | docker container prune
	yes | docker image prune
	yes | docker volume prune
	yes | docker network prune
	yes | docker system prune
	
	docker ps --all
	docker images
	docker system df
	docker volume ls
	docker network ls

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
