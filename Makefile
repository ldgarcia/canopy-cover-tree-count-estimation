test:
	#python -m unittest
	pytest tests --color=yes

clean:
	git clean -f -d -X;
