DATAFILE ?= datafile.dat
COMPILE ?= 0
FUNC_NAME ?= 

.PHONY: run
run:
	python3 make/run.py $(DATAFILE) $(COMPILE) $(FUNC_NAME)

dev:
	cargo run

release:
	cargo run --release

test:
	python3 make/test.py $(COMPILE) $(FUNC_NAME)

clean:
	rm make/data_comp.dat plotting/data.dat src/function.rs
