# Contributing

Thanks for your interest in improving the course. This project welcomes a few specific
kinds of contributions:

- **Corrections.** Mistakes in the text, broken cells, wrong formulas, stale APIs.
- **Additional exercises.** Open-ended problems that fit the difficulty curve of the week
  they belong to. Include a reference solution.
- **Alternative datasets.** Especially ones that are smaller, more permissively licensed,
  or more representative.
- **Translations.** Of either the prose or the notebook narration.

## Workflow

1. Open an issue first if the change is non-trivial — it's easier to discuss scope
   before code than after.
2. Fork the repo and create a topic branch: `git checkout -b fix/week3-roc-axis`.
3. Make your change. Keep notebook diffs reviewable: clear all outputs before committing,
   unless an output is the point of the cell.
4. Run the notebook end-to-end and make sure it executes top to bottom on a fresh kernel.
5. Open a pull request describing what changed and why.

## Style

- Notebooks are written in **English**.
- Code follows PEP 8; we use `ruff` for linting (`pip install ruff && ruff check .`).
- Figures use the course-wide style defined in `assets/mplstyle/course.mplstyle`. If you
  need to deviate, do it in a single cell and call out the deviation in the surrounding
  prose.
- Cite sources for any non-trivial claim. Markdown footnotes are fine.

## Reviewing

PRs need one reviewer's approval to merge. Reviewers should run the affected notebooks
locally, not just read the diff — visualization bugs love to hide in rendered output.
