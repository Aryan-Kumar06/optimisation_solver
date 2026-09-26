"""
NLP front-end for optimisation_solver.
Converts a plain-English optimization problem description into a
standard fixed-format .mps file consumable by `optimsolver`.

This module never calls the Claude API directly -- callers inject an
`adapter(system_prompt, user_prompt) -> str` function (see
llm_adapter.py). This keeps the LLM call swappable and makes the whole
pipeline deterministically testable without network access.
"""

import json
import re
import argparse

from sympy import symbols, sympify, Poly
from sympy.core.sympify import SympifyError

from schema import validate_spec, validate_name

# ---------------------------------------------------------------------
# Model semantics -- explicit and documented (PR #9 review item 4)
# ---------------------------------------------------------------------
# This frontend does NOT assume a default lower bound of 0 for variables
# whose bounds are unspecified in the spec. "Unspecified" (null/None)
# means genuinely free, and is encoded explicitly in the MPS BOUNDS
# section (FR / MI / UP / LO), never silently coerced to [0, +inf).
#
# If nonnegativity is the intended modeling default for a given problem,
# that must be stated explicitly in the spec (lower_bound: 0) -- by the
# LLM extraction step or by whoever constructs the spec -- not injected
# implicitly by this code.

SYSTEM_PROMPT = """You convert optimization problem descriptions into JSON.
Output ONLY valid JSON, no markdown fences, no commentary.

Schema:
{
  "variables": [{"name": str, "type": "continuous"|"integer"|"binary",
                 "lower_bound": number|null, "upper_bound": number|null}],
  "objective": {"sense": "maximize"|"minimize", "expression": str},
  "constraints": [{"name": str, "expression": str,
                    "operator": "<="|">="|"=", "rhs": number}]
}

Rules:
- expression fields must be linear (no products of two variables, no powers).
- Use variable names exactly as declared in "variables".
- Variable and constraint names must be valid identifiers, <= 8 characters
  (fixed-format MPS limit): start with a letter, then letters/digits/underscore only.
- lower_bound / upper_bound must be null if the problem does not state them.
  Do NOT assume 0 as a default lower bound -- only set it explicitly if the
  problem states or clearly implies nonnegativity.
- Give every constraint a short unique name like "c1", "c2".
"""


# ---------------------------------------------------------------------
# STEP 1: Natural language -> structured JSON spec (via injected adapter)
# ---------------------------------------------------------------------

def parse_nl_to_spec(problem_text: str, adapter) -> dict:
    """`adapter`: a function(system_prompt, user_prompt) -> str.
    See llm_adapter.py for the real Claude adapter and a canned test adapter."""
    raw = adapter(SYSTEM_PROMPT, problem_text)
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw output:\n{raw}")

    # Fail closed on malformed/incomplete output (PR #9 review item 3).
    validate_spec(spec)
    return spec


# ---------------------------------------------------------------------
# STEP 2: Validate expressions + extract linear coefficients with sympy
# ---------------------------------------------------------------------

def validate_and_extract(spec: dict):
    var_names = [v["name"] for v in spec["variables"]]

    # Re-validate names right before they touch sympy/MPS output
    # (PR #9 review item 5) -- defends this function even if it's ever
    # called with a spec that bypassed validate_spec().
    for name in var_names:
        validate_name(name, "variable")

    sym_map = {name: symbols(name) for name in var_names}

    def linear_coeffs(expr_str, context_label):
        try:
            expr = sympify(expr_str, locals=sym_map)
        except SympifyError as e:
            raise ValueError(f"Could not parse expression in {context_label}: '{expr_str}' ({e})")

        unknown = expr.free_symbols - set(sym_map.values())
        if unknown:
            raise ValueError(f"{context_label} references undeclared variable(s): {unknown}")

        try:
            poly = Poly(expr, *sym_map.values())
        except Exception as e:
            raise ValueError(f"Expression in {context_label} is not polynomial: {e}")

        if poly.total_degree() > 1:
            raise ValueError(
                f"Expression in {context_label} is nonlinear (degree {poly.total_degree()}). "
                f"This writer only supports LP/MILP (linear) models."
            )

        coeffs = {}
        for name, sym in sym_map.items():
            c = poly.coeff_monomial(sym) if sym in poly.gens else 0
            if c != 0:
                coeffs[name] = float(c)
        return coeffs

    obj_coeffs = linear_coeffs(spec["objective"]["expression"], "objective")

    constraints = []
    for c in spec["constraints"]:
        coeffs = linear_coeffs(c["expression"], f"constraint '{c['name']}'")
        if not coeffs:
            raise ValueError(f"constraint '{c['name']}' has no variable terms after parsing")
        constraints.append(
            {
                "name": c["name"],
                "coeffs": coeffs,
                "operator": c["operator"],
                "rhs": float(c["rhs"]),
            }
        )

    return obj_coeffs, constraints


# ---------------------------------------------------------------------
# STEP 3: Write fixed-format MPS file
# ---------------------------------------------------------------------

_ROW_TYPE = {"<=": "L", ">=": "G", "=": "E"}


def _bound_lines(name, vtype, lb, ub):
    """
    Explicit bound encoding -- never assumes a default (review item 4).
      - both None      -> FR (fully free: -inf, +inf)
      - lb only        -> LO  (MPS convention: ub stays +inf)
      - ub only        -> MI then UP (lower = -inf, explicit upper)
      - both given     -> LO, UP
      - binary         -> LO 0 / UP 1 explicitly, regardless of null
                          inputs, since a binary variable's domain is
                          {0,1} by definition of its declared type.
    """
    lines = []
    if vtype == "binary":
        effective_lb = 0 if lb is None else lb
        effective_ub = 1 if ub is None else ub
        lines.append(f" LO BND       {name:<10}{effective_lb:<12g}")
        lines.append(f" UP BND       {name:<10}{effective_ub:<12g}")
        return lines

    if lb is None and ub is None:
        lines.append(f" FR BND       {name}")
    elif lb is not None and ub is None:
        lines.append(f" LO BND       {name:<10}{lb:<12g}")
    elif lb is None and ub is not None:
        lines.append(f" MI BND       {name}")
        lines.append(f" UP BND       {name:<10}{ub:<12g}")
    else:
        lines.append(f" LO BND       {name:<10}{lb:<12g}")
        lines.append(f" UP BND       {name:<10}{ub:<12g}")
    return lines


def write_mps(spec: dict, obj_coeffs: dict, constraints: list, filepath: str,
              problem_name: str = "NLP_GENERATED"):
    variables = spec["variables"]
    var_names = [v["name"] for v in variables]
    var_type = {v["name"]: v["type"] for v in variables}
    sense = spec["objective"]["sense"]

    # MPS has no native "maximize" flag; convention here is to negate
    # the objective coefficients so the solver's minimize matches intent.
    sign = -1.0 if sense == "maximize" else 1.0

    lines = []
    lines.append(f"NAME          {problem_name}")
    lines.append("ROWS")
    lines.append(" N  COST")
    for c in constraints:
        lines.append(f" {_ROW_TYPE[c['operator']]}  {c['name']}")

    lines.append("COLUMNS")
    marker_count = 0
    in_int_block = False

    for vname in var_names:
        is_int = var_type[vname] in ("integer", "binary")

        # CRITICAL FIX (PR #9 review item 1): PL does not declare
        # integrality. Integer/binary status in MPS is conveyed only by
        # wrapping the variable's COLUMNS entries in INTORG/INTEND
        # markers -- so that's what we emit here.
        if is_int and not in_int_block:
            marker_count += 1
            lines.append(f"    MARKER                 'MARKER'                 'INTORG'")
            in_int_block = True
        elif not is_int and in_int_block:
            marker_count += 1
            lines.append(f"    MARKER                 'MARKER'                 'INTEND'")
            in_int_block = False

        entries = []
        if vname in obj_coeffs:
            entries.append(("COST", sign * obj_coeffs[vname]))
        for c in constraints:
            if vname in c["coeffs"]:
                entries.append((c["name"], c["coeffs"][vname]))

        if not entries:
            # Variable appears nowhere -- still needs a COLUMNS entry
            # (zero objective coefficient) so the solver registers it.
            entries.append(("COST", 0.0))

        for i in range(0, len(entries), 2):
            pair = entries[i:i + 2]
            parts = f"    {vname:<10}"
            for row, val in pair:
                parts += f"{row:<10}{val:<12g}"
            lines.append(parts.rstrip())

    if in_int_block:
        lines.append(f"    MARKER                 'MARKER'                 'INTEND'")

    lines.append("RHS")
    for c in constraints:
        lines.append(f"    RHS       {c['name']:<10}{c['rhs']:<12g}")

    lines.append("BOUNDS")
    for v in variables:
        lines.extend(_bound_lines(v["name"], v["type"], v.get("lower_bound"), v.get("upper_bound")))

    lines.append("ENDATA")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# STEP 4: End-to-end convenience function + CLI entry point
# ---------------------------------------------------------------------

def generate_mps_from_text(problem_text: str, adapter, output_path: str,
                            problem_name: str = "NLP_GENERATED"):
    spec = parse_nl_to_spec(problem_text, adapter)
    obj_coeffs, constraints = validate_and_extract(spec)
    write_mps(spec, obj_coeffs, constraints, output_path, problem_name)
    return spec


def main():
    parser = argparse.ArgumentParser(description="Convert NL problem description to .mps")
    parser.add_argument("problem_text", help="Plain-English optimization problem description")
    parser.add_argument("-o", "--output", default="generated.mps", help="Output .mps path")
    args = parser.parse_args()

    from llm_adapter import claude_adapter
    generate_mps_from_text(args.problem_text, claude_adapter, args.output)
    print(f"Wrote MPS file to {args.output}")


if __name__ == "__main__":
    main()