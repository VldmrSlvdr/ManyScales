.PHONY: check translation-check figures reports
check:
	python scripts/check.py

translation-check:
	Rscript translation/check.R

figures:
	Rscript scripts/redraw_figures.R

reports:
	python scripts/render_reports.py
