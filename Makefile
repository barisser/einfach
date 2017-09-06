test:
	export PYTHONPATH=.; py.test tests -vvv --cov einfach --cov-report=term-missing

lint:
	pylint einfach
