"""
NLP front-end for optimisation_solver.
Converts a plain-English optimization problem description into a
standard fixed-format .mps file consumable by `optimsolver`.
"""

import json
import re
import sys
import argparse
from sympy import symbols, sympify, Poly
from sympy.core.sympify import SympifyError

# ---------------------------------------------------------------------
# STEP 1: Natural language -> structured JSON spec (via Claude API)
# ---------------------------------------------------------------------

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
- If a bound is unspecified, use lower_bound: 0, upper_bound: null (i.e. x >= 0).
- Give every constraint a short unique name like "c1", "c2".
"""

def parse_nl_to_spec(problem_text: str, api_call_fn) -> dict:
    """
    api_call_fn: a function(system_prompt, user_prompt) -> str
    Kept as an injected dependency so you can plug in whichever
    HTTP client / SDK your project already uses for the Claude API.
    """
    raw = api_call_fn(SYSTEM_PROMPT, problem_text)
    raw = raw.strip()
    # strip accidental markdown fences defensively
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw output:\n{raw}")
    return spec


# ---------------------------------------------------------------------
# STEP 2: Validate spec + extract linear coefficients with sympy
# ---------------------------------------------------------------------

def validate_and_extract(spec: dict):
    var_names = [v["name"] for v in spec["variables"]]
    sym_map = {name: symbols(name) for name in var_names}

    def linear_coeffs(expr_str, context_label):
        try:
            expr = sympify(expr_str, locals=sym_map)
        except SympifyError as e:
            raise ValueError(f"Could not parse expression in {context_label}: '{expr_str}' ({e})")

        used_symbols = expr.free_symbols
        unknown = used_symbols - set(sym_map.values())
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
        constraints.append({
            "name": c["name"],
            "coeffs": coeffs,
            "operator": c["operator"],
            "rhs": float(c["rhs"]),
        })

    return obj_coeffs, constraints


# ---------------------------------------------------------------------
# STEP 3: Write fixed-format MPS file
# ---------------------------------------------------------------------

_ROW_TYPE = {"<=": "L", ">=": "G", "=": "E"}

def write_mps(spec: dict, obj_coeffs: dict, constraints: list, filepath: str,
              problem_name: str = "NLP_GENERATED"):
    variables = spec["variables"]
    var_names = [v["name"] for v in variables]
    sense = spec["objective"]["sense"]

    # MPS has no native "maximize" flag; convention here is to negate
    # the objective coefficients so the solver's minimize matches intent.
    # If your optimsolver CLI already supports a --maximize flag, prefer
    # that over negation and skip this step.
    sign = -1.0 if sense == "maximize" else 1.0

    lines = []
    lines.append(f"NAME          {problem_name}")
    lines.append("ROWS")
    lines.append(" N  COST")
    for c in constraints:
        lines.append(f" {_ROW_TYPE[c['operator']]}  {c['name']}")

    lines.append("COLUMNS")
    for vname in var_names:
        entries = []
        if vname in obj_coeffs:
            entries.append(("COST", sign * obj_coeffs[vname]))
        for c in constraints:
            if vname in c["coeffs"]:
                entries.append((c["name"], c["coeffs"][vname]))
        # emit two (row, value) pairs per line, MPS-style
        for i in range(0, len(entries), 2):
            pair = entries[i:i + 2]
            parts = f"    {vname:<10}"
            for row, val in pair:
                parts += f"{row:<10}{val:<12g}"
            lines.append(parts.rstrip())

    lines.append("RHS")
    for c in constraints:
        lines.append(f"    RHS       {c['name']:<10}{c['rhs']:<12g}")

    lines.append("BOUNDS")
    for v in variables:
        name = v["name"]
        lb = v.get("lower_bound")
        ub = v.get("upper_bound")
        if v["type"] == "binary":
            lines.append(f" BV BND       {name}")
            continue
        if lb is not None and lb != 0:
            lines.append(f" LO BND       {name:<10}{lb:<12g}")
        if ub is not None:
            lines.append(f" UP BND       {name:<10}{ub:<12g}")
        if v["type"] == "integer" and ub is None:
            lines.append(f" PL BND       {name}")  # explicit plus-infinity, keeps it integer-flagged upstream if your parser needs a bound line

    lines.append("ENDATA")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# STEP 4: CLI entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert NL problem description to .mps")
    parser.add_argument("problem_text", help="Plain-English optimization problem description")
    parser.add_argument("-o", "--output", default="generated.mps", help="Output .mps path")
    args = parser.parse_args()

    # Plug in your actual Claude API call here.
    def call_claude(system_prompt, user_prompt):
        raise NotImplementedError(
            "Wire this up to your Claude API client (see step-by-step notes)."
        )

    spec = parse_nl_to_spec(args.problem_text, call_claude)
    obj_coeffs, constraints = validate_and_extract(spec)
    write_mps(spec, obj_coeffs, constraints, args.output)
    print(f"Wrote MPS file to {args.output}")


if __name__ == "__main__":
    main()