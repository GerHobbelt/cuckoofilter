.PHONY: clean continuous

main.pdf: Makefile main.tex
	latexmk -pdf main.tex

clean:
	latexmk -C main.tex

continuous:
	yes x | latexmk -pdf -pvc main.tex
