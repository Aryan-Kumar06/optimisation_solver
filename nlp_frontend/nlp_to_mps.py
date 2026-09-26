"""
Test suite for the NLP frontend (nlp_frontend/nlp_to_mps.py).

All LLM calls are mocked via injected canned adapters
(llm_adapter.make_canned_adapter) -- no network access, no API credits
consumed anywhere in this file.

End-to-end tests use the `optimsolver_path` fixture from conftest.py,
which hard-fails (not skips) if no binary can be found or built.
"""

import os
import re
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

SINGLE_VAR_LE = """
{
  "variables": [{"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": null}],
  "objective": {"sense": "maximize", "expression": "x"},
  "constraints": [{"name": "c1", "expression": "x + 5", "operator": "<=", "rhs": 10}]
}
"""


# ---------------------------------------------------------------------
# Parsing / extraction happy-path tests
# ---------------------------------------------------------------------

def test_continuous_lp_parses_and_extracts():
    spec = parse_nl_to_spec("dummy", make_canned_adapter(CONTINUOUS_LP))
    obj_coeffs, obj_const, constraints = validate_and_extract(spec)
    assert obj_coeffs == {"x": 3.0, "y": 5.0}
    assert obj_const == 0.0
    assert constraints[0]["coeffs"] == {"x": 1.0, "y": 2.0}
    assert constraints[0]["rhs"] == 100.0


def test_integer_lp_type_preserved():
    spec = parse_nl_to_spec("dummy", make_canned_adapter(INTEGER_LP))
    assert spec["variables"][0]["type"] == "integer"


def test_binary_lp_type_preserved():
    spec = parse_nl_to_spec("dummy", make_canned_adapter(BINARY_LP))
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
    obj_coeffs, _, _ = validate_and_extract(spec)
    assert obj_coeffs["x"] == -3.0


def test_bounds_not_silently_defaulted_to_zero():
    spec = parse_nl_to_spec("dummy", make_canned_adapter(BINARY_LP))
    v = spec["variables"][0]
    assert v["lower_bound"] is None
    assert v["upper_bound"] is None


# ---------------------------------------------------------------------
# Constant-term regression tests (CRITICAL fix, review round 2 item 1)
# ---------------------------------------------------------------------

def test_constant_folded_into_rhs_le():
    """x + 5 <= 10  =>  x <= 5"""
    spec = parse_nl_to_spec("dummy", make_canned_adapter(SINGLE_VAR_LE))
    obj_coeffs, obj_const, constraints = validate_and_extract(spec)
    assert constraints[0]["coeffs"] == {"x": 1.0}
    assert constraints[0]["rhs"] == 5.0


def test_constant_folded_into_rhs_ge():
    """x - 5 >= 10  =>  x >= 15"""
    canned = SINGLE_VAR_LE.replace('"x + 5"', '"x - 5"').replace('"<="', '">="')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    _, _, constraints = validate_and_extract(spec)
    assert constraints[0]["rhs"] == 15.0


def test_constant_folded_into_rhs_eq():
    """x + 5 = 10  =>  x = 5"""
    canned = SINGLE_VAR_LE.replace('"<="', '"="')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    _, _, constraints = validate_and_extract(spec)
    assert constraints[0]["rhs"] == 5.0


def test_negative_constant_via_unary_minus():
    """x + -5 <= 10  =>  x <= 15"""
    canned = SINGLE_VAR_LE.replace('"x + 5"', '"x + -5"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    _, _, constraints = validate_and_extract(spec)
    assert constraints[0]["rhs"] == 15.0


def test_objective_constant_preserved():
    """maximize 3x + 5y + 7  ->  coeffs unchanged, constant = 7 (not dropped)"""
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"3*x + 5*y + 7"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    obj_coeffs, obj_const, _ = validate_and_extract(spec)
    assert obj_coeffs == {"x": 3.0, "y": 5.0}
    assert obj_const == 7.0


def test_objective_constant_written_to_mps(tmp_path):
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"3*x + 5*y + 7"')
    out = tmp_path / "constobj.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(out))
    content = out.read_text()
    assert "RHS       COST" in content


def test_zero_objective_constant_omits_rhs_line(tmp_path):
    out = tmp_path / "nooffset.mps"
    generate_mps_from_text("dummy", make_canned_adapter(CONTINUOUS_LP), str(out))
    content = out.read_text()
    assert "RHS       COST" not in content


# ---------------------------------------------------------------------
# Safe-parser security / robustness tests (review round 2 item 2)
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


def test_exponentiation_rejected():
    """Powers are outside the grammar -- must be rejected, not silently evaluated."""
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"3*x**2 + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError):
        validate_and_extract(spec)


def test_function_call_rejected():
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"abs(x) + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError):
        validate_and_extract(spec)


def test_attribute_access_rejected():
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"x.bit_length() + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError):
        validate_and_extract(spec)


def test_call_expression_rejected_before_evaluation():
    """Demonstrates the trust boundary: the parser never executes
    anything -- a Call node is rejected by ast inspection alone,
    regardless of what it names."""
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"os.system(1) + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError):
        validate_and_extract(spec)


def test_variable_division_by_variable_rejected():
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"x/y + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    with pytest.raises(ValueError, match="nonlinear"):
        validate_and_extract(spec)


def test_division_by_constant_allowed():
    """x/2 is still linear and should be accepted."""
    canned = CONTINUOUS_LP.replace('"3*x + 5*y"', '"x/2 + 5*y"')
    spec = parse_nl_to_spec("dummy", make_canned_adapter(canned))
    obj_coeffs, _, _ = validate_and_extract(spec)
    assert obj_coeffs["x"] == 0.5


# ---------------------------------------------------------------------
# Schema failure-path tests
# ---------------------------------------------------------------------

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
        '"lower_bound": 0, "upper_bound": null', '"lower_bound": 50, "upper_bound": 10', 1
    )
    with pytest.raises(SchemaValidationError, match="lower_bound"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


# ---------------------------------------------------------------------
# Finite-number validation tests (review round 2 item 3)
# ---------------------------------------------------------------------

def test_nan_rhs_rejected():
    canned = CONTINUOUS_LP.replace('"rhs": 100', '"rhs": NaN')
    with pytest.raises(SchemaValidationError, match="finite"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_infinite_lower_bound_rejected():
    canned = CONTINUOUS_LP.replace('"lower_bound": 0', '"lower_bound": Infinity', 1)
    with pytest.raises(SchemaValidationError, match="finite"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


def test_negative_infinite_rhs_rejected():
    canned = CONTINUOUS_LP.replace('"rhs": 100', '"rhs": -Infinity')
    with pytest.raises(SchemaValidationError, match="finite"):
        parse_nl_to_spec("dummy", make_canned_adapter(canned))


# ---------------------------------------------------------------------
# MPS writer / integer encoding tests
# ---------------------------------------------------------------------

def _line_starting_with(content, token):
    for i, l in enumerate(content.splitlines()):
        if l.strip().split()[:1] == [token]:
            return i
    return None


def test_integer_variable_gets_intorg_intend_markers(tmp_path):
    out = tmp_path / "int.mps"
    generate_mps_from_text("dummy", make_canned_adapter(INTEGER_LP), str(out))
    content = out.read_text()
    assert "INTORG" in content and "INTEND" in content
    lines = content.splitlines()
    intorg_idx = next(i for i, l in enumerate(lines) if "INTORG" in l)
    var_idx = _line_starting_with(content, "x")
    assert intorg_idx < var_idx


def test_continuous_variable_not_wrapped_in_markers(tmp_path):
    out = tmp_path / "cont.mps"
    generate_mps_from_text("dummy", make_canned_adapter(CONTINUOUS_LP), str(out))
    content = out.read_text()
    assert "INTORG" not in content and "INTEND" not in content


def test_binary_variable_wrapped_and_bounded(tmp_path):
    out = tmp_path / "bin.mps"
    generate_mps_from_text("dummy", make_canned_adapter(BINARY_LP), str(out))
    content = out.read_text()
    assert "INTORG" in content and "INTEND" in content
    assert re.search(r"LO BND\s+b1\s+0", content)
    assert re.search(r"UP BND\s+b1\s+1", content)


def test_free_bounds_use_fr_not_default_zero(tmp_path):
    canned = CONTINUOUS_LP.replace(
        '"lower_bound": 0, "upper_bound": null', '"lower_bound": null, "upper_bound": null', 1
    )
    out = tmp_path / "free.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(out))
    assert "FR BND" in out.read_text()


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
    n_idx = _line_starting_with(content, "n")
    x_idx = _line_starting_with(content, "x")
    y_idx = _line_starting_with(content, "y")
    assert intorg_idx < n_idx < intend_idx
    assert not (intorg_idx < x_idx < intend_idx)
    assert not (intorg_idx < y_idx < intend_idx)


# ---------------------------------------------------------------------
# End-to-end: generated MPS actually solved by optimsolver
#
# These use the `optimsolver_path` fixture (conftest.py), which HARD
# FAILS (not skips) if no binary is available -- review round 2 item 4.
#
# NOTE on _extract_objective(): this regex is a best-effort guess at
# optimsolver's CLI output format and has NOT been verified against a
# real run (review round 2 item 4 asks for this verification). Update
# it once the actual output format is confirmed.
# ---------------------------------------------------------------------

def _extract_objective(stdout: str) -> float:
    patterns = [
        r"[Oo]bjective(?:\s+value)?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"[Oo]ptimal\s+objective\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
        r"obj\s*=\s*(-?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, stdout)
        if m:
            return float(m.group(1))
    raise AssertionError(f"Could not find objective value in optimsolver output:\n{stdout}")


def _solve(binary, mps_path):
    result = subprocess.run([binary, "solve", str(mps_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"optimsolver failed:\n{result.stderr}"
    return _extract_objective(result.stdout)


def test_e2e_continuous_lp_matches_expected_optimum(optimsolver_path, tmp_path):
    """maximize 3x+5y, x+2y<=100, x<=40 -> optimum at x=40,y=30, obj=420."""
    mps_path = tmp_path / "cont_e2e.mps"
    generate_mps_from_text("dummy", make_canned_adapter(CONTINUOUS_LP), str(mps_path))
    assert abs(_solve(optimsolver_path, mps_path) - 420.0) < 1e-3


def test_e2e_integer_rounds_down(optimsolver_path, tmp_path):
    """
    maximize x, x integer, 0 <= x <= 7.5.
    Expected optimum x=7 (not 7.5) -- proves the generated MPS is
    actually solved as an integer program, not just LP-relaxed.
    """
    canned = """
    {
      "variables": [{"name": "x", "type": "integer", "lower_bound": 0, "upper_bound": 7.5}],
      "objective": {"sense": "maximize", "expression": "x"},
      "constraints": []
    }
    """
    mps_path = tmp_path / "int_e2e.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(mps_path))
    assert abs(_solve(optimsolver_path, mps_path) - 7.0) < 1e-6


def test_e2e_binary_binding(optimsolver_path, tmp_path):
    """maximize 3*b1+4*b2, b1+b2<=1, both binary -> optimum picks b2, obj=4."""
    mps_path = tmp_path / "bin_e2e.mps"
    generate_mps_from_text("dummy", make_canned_adapter(BINARY_LP), str(mps_path))
    assert abs(_solve(optimsolver_path, mps_path) - 4.0) < 1e-6


def test_e2e_mixed_continuous_integer(optimsolver_path, tmp_path):
    """
    Separable model so the optimum is easy to predict by hand, while
    still exercising INTORG/INTEND boundaries with continuous variables
    both before and after the integer block:
        maximize x + n + y ; x<=3 (continuous), n<=5.5 (integer), y<=4 (continuous)
    Expected optimum: x=3, n=5, y=4, objective=12.
    """
    canned = """
    {
      "variables": [
        {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": 3},
        {"name": "n", "type": "integer", "lower_bound": 0, "upper_bound": 5.5},
        {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": 4}
      ],
      "objective": {"sense": "maximize", "expression": "x + n + y"},
      "constraints": []
    }
    """
    mps_path = tmp_path / "mixed_e2e.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(mps_path))
    assert abs(_solve(optimsolver_path, mps_path) - 12.0) < 1e-6


def test_e2e_maximize_reports_correct_sign(optimsolver_path, tmp_path):
    """maximize 3x+2y, x+y<=4 -> optimum x=4,y=0, obj=12 (proves the internal
    objective negation for MPS is correctly un-done when reading the result)."""
    canned = """
    {
      "variables": [
        {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": null},
        {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": null}
      ],
      "objective": {"sense": "maximize", "expression": "3*x + 2*y"},
      "constraints": [{"name": "c1", "expression": "x + y", "operator": "<=", "rhs": 4}]
    }
    """
    mps_path = tmp_path / "max_e2e.mps"
    generate_mps_from_text("dummy", make_canned_adapter(canned), str(mps_path))
    assert abs(_solve(optimsolver_path, mps_path) - 12.0) < 1e-6