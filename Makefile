CARGO_LDFLAGS ?=
CARGO_TARGET ?= debug
ifeq ($(RELEASE),1)
	override CARGO_OPTS += --release
	CARGO_TARGET = release
endif

CLIPPY_OPTS ?=

SILENCE = @
ifeq ($(VERBOSE),1)
	SILENCE =
endif

.PHONY: build
build:
	@echo "Building OmegaOptimizer"
	cargo build $(CARGO_OPTS)
	cp target/$(CARGO_TARGET)/omega_optimizer .

.PHONY: clean
clean:
	cargo clean

.PHONY: format
format:
	@echo "Formatting files"
	@# I can't use cargo fmt as the files in 'functions' are mod-ed in a macro
	$(SILENCE)rustfmt src/main.rs src/functions/*.rs --edition=2024

.PHONY: lint
lint:
	@echo "Linting"
	$(SILENCE)cargo clippy $(CARGO_OPTS) --quiet -- $(CLIPPY_OPTS)

.PHONY: check
check:
	@echo "Running tests"
	$(SILENCE)cargo test $(CARGO_OPTS) --quiet

.PHONY: update
update:
	cargo +nightly update

.PHONY: doc
doc:
	cargo doc --document-private-items --no-deps

.PHONY: ci
ci:
	$(SILENCE)MAKEFLAGS=--no-print-directory make format
	$(SILENCE)MAKEFLAGS=--no-print-directory make lint CLIPPY_OPTS="-D warnings"
	$(SILENCE)MAKEFLAGS=--no-print-directory make check
