
.PHONY: run
run:
	python3 make/run.py

build:
	cargo run

test:
	python3 make/test.py