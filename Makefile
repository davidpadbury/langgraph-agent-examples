.PHONY: all
all: install lint type-check

.PHONY: lint
lint:
	ruff check

.PHONY: mypy
mypy:
	mypy src

.PHONY: install
install:
	pip install -r requirements.txt
	pip install -e .

.PHONY: freeze
freeze:
	rm requirements.txt
	pip freeze | grep -v "^-e" > requirements.txt
	echo "-e ." >> requirements.txt

.PHONY: test-examples
test-examples: install
	scripts/test-examples.sh