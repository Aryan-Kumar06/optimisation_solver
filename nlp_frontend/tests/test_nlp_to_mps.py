"""
Test suite for the NLP frontend (nlp_frontend/nlp_to_mps.py).

All LLM calls are mocked via injected canned adapters (llm_adapter.make_
canned_adapter) -- no network access, no API credits consumed anywhere
in this file. Addresses PR #9 review item 7.

Covers (review item 6): continuous LP, integer LP, binary LP, min/max,
<=/>=/=, bounds, negative coefficients, nonlinear expressions, unknown
variables, malformed LLM JSON, and an end-to-end generated-MPS ->
optimsolver check (review item 8).
"""

import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_to_mps import parse_nl_to_spec, validate_and_extract, generate_mps_from_text
from schema import SchemaValidationError
from llm_adapter import make_canned_adapter


# ---------------------------------------------------------------------
# Canned LLM responses (fixtures)
# ---------------------------------------------------------------------

CONTINUOUS_LP = """
{
  "variables": [
    {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": null},
    {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": null}
  ],
  "objective": {"sense": "maximize", "expression": "3*x + 5*y"},
  "constraints": [
    {"name": "c1", "expression": "x + 2*y", "operator": "<=", "rhs": 100},
    {"name": "c2", "expression": "x", "operator": "<=", "rhs": 40}
  ]
}
"""

INTEGER_LP = """
{
  "variables": [
    {"name": "x", "type": "integer", "lower_bound": 0, "upper_bound": 10}
  ],
  "objective": {"sense": "maximize", "expression": "x"},
  "constraints": [
    {"name": "c1", "expression": "x", "operator": "<=", "rhs": 7}
  ]
}
"""

BINARY_LP = """
{
  "variables": [
    {"name": "b1", "type": "binary", "lower_bound": null, "upper_bound": null},
    {"name": "b2", "type": "binary", "lower_bound": null, "upper_bound": null}
  ],
  "objective": {"sense": "maximize", "expression": "3*b1 + 4*b2"},
  "constraints": [
    {"name": "c1", "expression": "b1 + b2", "operator": "<=", "rhs": 1}
  ]
}
"""


# ---------------------------------------------------------------------
# Parsing / extraction happy-path tests
# ---------------------------------------------------------------------

def test_continuous_lp_parses_and_extracts():
    adapter = make_canned_adapter(CONTINUOUS_LP)
    spec = parse_nl_to_spec("dummy prompt", adapter)
    obj_coeffs, constraints = validate_and_extract(spec)
    assert obj_coeffs == {"x": 3.0, "y": 5.0}
    assert constraints[0]["coeffs"] == {"x": 1.0, "y": 2.0}
    assert constraints[0]["operator"] == "<="
    assert constraints[0]["rhs"] == 100.0


def test_integer_lp_type_preserved():
    adapter = make_canned_adapter(INTEGER_LP)
    spec = parse_nl_to_spec("dummy prompt", adapter)
    assert spec["variables"][0]["type"] == "integer"


def test_binary_lp_type_preserved():
    adapter = make_canned_adapter(BINARY_LP)
    spec = parse_nl_to_spec("dummy prompt", adapter)
    assert [v["type"] for v in spec["variables"]] == ["binary", "binary"]


def test_min_sense():
    canned = CONTINUOUS_LP.replace('"maximize"', '"minimize"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    assert spec["objective"]["sense"] == "minimize"


@pytest.mark.parametrize("operator", ["<=", ">=", "="])
def test_all_operators_accepted(operator):
    canned = CONTINUOUS_LP.replace('"<="', f'"{operator}"', 1)
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    assert spec["constraints"][0]["operator"] == operator


def test_negative_coefficients():
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"-3*x + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    obj_coeffs, _ = validate_and_extract(spec)
    assert obj_coeffs["x"] == -3.0


def test_bounds_not_silently_defaulted_to_zero():
    """Unspecified bounds must remain None, never become an implicit 0."""
    spec = parse_nl_to_spec("dummy", make_canned_adapter(BINARY_LP))
    v = spec["variables"][0]
    assert v["lower_bound"] is None
    assert v["upper_bound"] is None


# ---------------------------------------------------------------------
# Failure-path tests -- fail closed on malformed/invalid input
# ---------------------------------------------------------------------

def test_nonlinear_expression_rejected():
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"3*x*y + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError, match="nonlinear"):
        validate_and_extract(spec)


def test_unknown_variable_rejected():
    canned = CONTINUOUS_LP.replace('"x + 2*y"', '"x + 2*z"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError, match="undeclared"):
        validate_and_extract(spec)


def test_malformed_llm_json_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_nl_to_spec("dummy", make_canned_adapter("{not valid json,,,"))


def test_missing_required_key_fails_closed():
    canned = '{"variables": [{"name": "x", "type": "continuous"}], "objective": {"sense": "maximize", "expression": "x"}}'
    with pytest.raises(SchemaValidationError, match="constraints"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_empty_variables_rejected():
    canned = '{"variables": [], "objective": {"sense": "maximize", "expression": "x"}, "constraints": []}'
    with pytest.raises(SchemaValidationError):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_invalid_variable_name_rejected():
    canned = CONTINUOUS_LP.replace('"name": "x"', '"name": "9x"')
    with pytest.raises(SchemaValidationError, match="not a valid identifier"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_overlong_variable_name_rejected():
    canned = CONTINUOUS_LP.replace('"name": "x"', '"name": "waytoolongname"')
    with pytest.raises(SchemaValidationError, match="exceeds"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_duplicate_variable_names_rejected():
    canned = CONTINUOUS_LP.replace('"name": "y"', '"name": "x"')
    with pytest.raises(SchemaValidationError, match="duplicate"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_bad_operator_rejected():
    canned = CONTINUOUS_LP.replace('"<="', '"!="', 1)
    with pytest.raises(SchemaValidationError, match="operator"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_non_numeric_rhs_rejected():
    canned = CONTINUOUS_LP.replace('"rhs": 100', '"rhs": "one hundred"')
    with pytest.raises(SchemaValidationError, match="rhs"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_inconsistent_bounds_rejected():
    canned = CONTINUOUS_LP.replace(
        '"lower_bound": 0, "upper_bound": null',
        '"lower_bound": 50, "upper_bound": 10',
        1,
    )
    with pytest.raises(SchemaValidationError, match="lower_bound"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


# ---------------------------------------------------------------------
# MPS writer / integer encoding tests (review item 1 -- the critical fix)
# ---------------------------------------------------------------------

def _line_starting_with(content, token):
    for line in content.splitlines():
        if line.strip().split()[:1] == [token]:
            return line
    return None


def test_integer_variable_gets_intorg_intend_markers(tmp_path):
    out = tmp_path / "int.mps"
    generate_mps_from_text("dummy", make_canned_adapter(INTEGER_LP), str(out))
    content = out.read_text()
    assert "INTORG" in content
    assert "INTEND" in content

    lines = content.splitlines()
    intorg_idx = next(i for i, l in enumerate(lines) if "INTORG" in l)
    var_idx = next(i for i, l in enumerate(lines) if l.strip().split()[:1] == ["x"])
    assert intorg_idx < var_idx, "INTORG marker must precede the integer variable's COLUMNS entry"


def test_continuous_variable_not_wrapped_in_markers(tmp_path):
    out = tmp_path / "cont.mps"
    generate_mps_from_text("dummy", make_canned_adapter(CONTINUOUS_LP), str(out))
    content = out.read_text()
    assert "INTORG" not in content
    assert "INTEND" not in content


def test_binary_variable_wrapped_and_bounded(tmp_path):
    out = tmp_path / "bin.mps"
    generate_mps_from_text("dummy", make_canned_adapter(BINARY_LP), str(out))
    content = out.read_text()
    assert "INTORG" in content and "INTEND" in content
    assert re.search(r"LO BND\s+b1\s+0", content)
    assert re.search(r"UP BND\s+b1\s+1", content)


def test_free_bounds_use_fr_not_default_zero(tmp_path):
    """A variable with both bounds unspecified must be written as FR
    (free), never silently as an implicit LO 0."""
    canned = CONTINUOUS_LP.replace(
        '"lower_bound": 0, "upper_bound": null', '"lower_bound": null, "upper_bound": null', 1
    )
    out = tmp_path / "free.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(out))
    content = out.read_text()
    assert "FR BND" in content


def test_mixed_integer_and_continuous_only_wraps_integer_block(tmp_path):
    canned = """
    {
      "variables": [
        {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": null},
        {"name": "n", "type": "integer", "lower_bound": 0, "upper_bound": 5},
        {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": null}
      ],
      "objective": {"sense": "maximize", "expression": "x + n + y"},
      "constraints": [
        {"name": "c1", "expression": "x + n + y", "operator": "<=", "rhs": 20}
      ]
    }
    """
    out = tmp_path / "mixed.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(out))
    content = out.read_text()
    assert content.count("INTORG") == 1
    assert content.count("INTEND") == 1
    lines = content.splitlines()
    intorg_idx = next(i for i, l in enumerate(lines) if "INTORG" in l)
    intend_idx = next(i for i, l in enumerate(lines) if "INTEND" in l)
    n_idx = next(i for i, l in enumerate(lines) if l.strip().split()[:1] == ["n"])
    x_idx = next(i for i, l in enumerate(lines) if l.strip().split()[:1] == ["x"])
    y_idx = next(i for i, l in enumerate(lines) if l.strip().split()[:1] == ["y"])
    assert intorg_idx < n_idx < intend_idx
    assert not (intorg_idx < x_idx < intend_idx)
    assert not (intorg_idx < y_idx < intend_idx)


# ---------------------------------------------------------------------
# End-to-end: generated MPS actually solved by optimsolver
# (review item 8 -- and mandatory, per the review, since correctness of
# solver input is the entire point of this frontend)
# ---------------------------------------------------------------------

def _find_optimsolver():
    """Look in PATH and the conventional build/ directory; skip the
    end-to-end test if no built binary is available."""
    candidates = [
        shutil.which("optimsolver"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "build",
            "optimsolver",
        ),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


@pytest.mark.skipif(_find_optimsolver() is None, reason="optimsolver binary not found; build it first")
def test_end_to_end_generated_mps_matches_expected_optimum(tmp_path):
    """
    maximize 3x + 5y
    s.t.     x + 2y <= 100
             x      <= 40
             x, y   >= 0
    Known optimum: x=40, y=30, objective=420
    (dominates the (0, 50) vertex: 5*50=250 < 420).

    NOTE: the objective-value regex below assumes optimsolver prints a
    line containing "objective" followed by a number. Update this
    pattern to match the CLI's actual output format before relying on
    this test in CI.
    """
    binary = _find_optimsolver()
    mps_path = tmp_path / "e2e.mps"
    generate_mps_from_text("maximize 3x+5y s.t. x+2y<=100, x<=40", make_canned_adapter(CONTINUOUS_LP), str(mps_path))

    result = subprocess.run([binary, "solve", str(mps_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"optimsolver failed:\n{result.stderr}"

    match = re.search(r"[Oo]bjective[^0-9\-]*(-?\d+(?:\.\d+)?)", result.stdout)
    assert match, f"Could not find objective value in output:\n{result.stdout}"
    objective = float(match.group(1))
    assert abs(objective - 420.0) < 1e-3, f"Expected objective 420, got {objective}"