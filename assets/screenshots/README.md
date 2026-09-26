# Screenshots

This directory is intended to store terminal captures and CLI screenshots for documentation and the SIH 2026 presentation.

---

## Planned Screenshots Catalogue

The following terminal captures are planned before final submission:

| Filename | Command / Interaction | Purpose |
| :--- | :--- | :--- |
| `01-interactive-home.png` | `./build/optimsolver` | Terminal home screen with mascot banner and numbered main menu |
| `02-interactive-model-summary.png` | Option `1` with `tests/cli/simple_lp.mps` | Structured model information card showing dimensions, non-zeros, objective terms, and detected problem class |
| `03-interactive-solve-result.png` | Option `1` solve execution | Solve dashboard card showing status (`Optimal`), objective value, iterations, and dual availability |
| `04-cli-solve-help.png` | `./build/optimsolver solve --help` | Command-line help screen displaying arguments, flags (`--solver`, `--time-limit`, `--output`), and usage examples |
| `05-cli-batch-solve.png` | `./build/optimsolver solve tests/cli/simple_lp.mps` | Non-interactive batch solve card on standard continuous LP |
| `06-presolve-reduction.png` | `./build/optimsolver solve tests/cli/presolve_reduction.mps --output /tmp/solution.txt` | Dimension reduction display (3 variables reduced to 2) and solution export confirmation |
| `07-ctest-execution.png` | `ctest --test-dir build --output-on-failure` | CTest execution summary showing 100% pass rate across the automated test suite |

---

## Guidelines

- Save captures in standard `.png` format.
- Use clean, high-contrast terminal captures with legible fonts.
- Maintain the leading two-digit sequence numbers above.
- Do not commit mock or placeholder images; add only real terminal captures.
