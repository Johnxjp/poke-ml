test:
	pytest tests

install:
	pip install -r requirements.txt

install-test:
	pip install -r test_requirements.txt

.PHONY: test, install, install-test