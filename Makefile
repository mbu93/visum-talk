.PHONY: all image push

all: image push

image:
	pipreqs --force src
	cp src/main.py docker/pipeline
	cp src/requirements.txt docker/pipeline
	docker build -t mbu93/visum-pipeline-runner:latest docker/pipeline

push:
	rm docker/pipeline/main.py
	rm docker/pipeline/requirements.txt
	docker push mbu93/visum-pipeline-runner:latest
