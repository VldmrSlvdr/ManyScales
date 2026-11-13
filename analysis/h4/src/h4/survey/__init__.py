"""Survey export reading, exclusion flags, questionnaire codebook, and reshape.

This layer reproduces the Quarto preprocessing in
``analysis/ratings/primary`` (citizen and expert exclusion flags, then the
set-level analysis tables). ``python -m h4.survey.preanalysis`` checks that
cut and writes the long analysis tables. ``python -m h4.survey.metaeval``
joins those tables to the stored metrics. ``python -m h4.survey.delta_pipeline``
is the study analysis: the coefficient of ``delta_z`` on the rating gap,
preference, and the LLM guess. ``python -m h4.survey.mixed_judgments`` fits
those models together with the expert item-flag checks. The cosine file
``results/tables/confirmatory_estimate.csv`` is the coauthor replicate and
is left unchanged.
"""
