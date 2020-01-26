.PHONY: clean continuous

main.pdf: Makefile main.tex
	latexmk -pdf main.tex

clean:
	latexmk -C main.tex
	rm main.bbl

continuous:
	yes x | latexmk -pdf -pvc main.tex
