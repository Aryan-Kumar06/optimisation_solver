"""
NLP front-end for optimisation_solver.
Converts a plain-English optimization problem description into a
standard fixed-format .mps file consumable by `optimsolver`.

This module never calls the Claude API directly -- callers inject an
`adapter(system_prompt, user_prompt) -> str` function (see
llm_adapter.py). This keeps the LLM call swappable and makes the whole
pipeline deterministically testable without network access.
"""

import argparse
import json
import math
import re

from safe_expr import linear_terms, ExpressionError
from schema import validate_spec, validate_name, SchemaValidationError

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
# STEP 2: Validate expressions + extract linear terms using safe_expr
# ---------------------------------------------------------------------

def validate_and_extract(spec: dict):
    """
    Validates variable names and expressions using the restricted, safe AST parser.
    Folds any constant terms in constraints into the RHS:
        lhs + const <= rhs  =>  lhs <= rhs - const
    Preserves any objective constant term so callers can track the full objective.

    Returns:
        obj_coeffs: dict[str, float]
        obj_const: float
        constraints: list[dict]
    """
    var_names = [v["name"] for v in spec["variables"]]

    # Re-validate names right before they touch MPS output
    for name in var_names:
        validate_name(name, "variable")

    var_set = set(var_names)

    # Objective extraction
    obj_coeffs, obj_const = linear_terms(spec["objective"]["expression"], var_set, "objective")

    constraints = []
    for c in spec["constraints"]:
        validate_name(c["name"], "constraint")
        coeffs, const = linear_terms(c["expression"], var_set, f"constraint '{c['name']}'")
        if not coeffs:
            raise ValueError(f"constraint '{c['name']}' has no variable terms after parsing")

        effective_rhs = float(c["rhs"]) - const
        if not math.isfinite(effective_rhs):
            raise ValueError(f"constraint '{c['name']}': folded rhs is not finite ({effective_rhs})")

        constraints.append(
            {
                "name": c["name"],
                "coeffs": coeffs,
                "operator": c["operator"],
                "rhs": effective_rhs,
            }
        )

    return obj_coeffs, obj_const, constraints


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


def write_mps(spec: dict, obj_coeffs: dict, *args, problem_name: str = "NLP_GENERATED", **kwargs):
    """
    Writes fixed-format MPS with OBJSENSE extension and INTORG/INTEND markers.
    Accepts either:
        write_mps(spec, obj_coeffs, constraints, filepath, problem_name)
    or:
        write_mps(spec, obj_coeffs, obj_const, constraints, filepath, problem_name)
    """
    if len(args) == 2:
        obj_const = 0.0
        constraints, filepath = args
    elif len(args) >= 3:
        if isinstance(args[0], (int, float)):
            obj_const = float(args[0])
            constraints = args[1]
            filepath = args[2]
            if len(args) >= 4:
                problem_name = args[3]
        else:
            obj_const = 0.0
            constraints = args[0]
            filepath = args[1]
            problem_name = args[2]
    else:
        raise TypeError("write_mps expects at least (spec, obj_coeffs, constraints, filepath)")

    variables = spec["variables"]
    var_names = [v["name"] for v in variables]
    var_type = {v["name"]: v["type"] for v in variables}
    sense = spec["objective"]["sense"]

    lines = []
    lines.append(f"NAME          {problem_name}")

    # Standard OBJSENSE extension natively supported by optimsolver and modern readers
    lines.append("OBJSENSE")
    if sense == "maximize":
        lines.append(" MAX")
    else:
        lines.append(" MIN")

    lines.append("ROWS")
    lines.append(" N  COST")
    for c in constraints:
        lines.append(f" {_ROW_TYPE[c['operator']]}  {c['name']}")

    lines.append("COLUMNS")
    in_int_block = False

    for vname in var_names:
        is_int = var_type[vname] in ("integer", "binary")

        # Integer/binary status in MPS is conveyed by INTORG/INTEND markers
        if is_int and not in_int_block:
            lines.append("    MARKER                 'MARKER'                 'INTORG'")
            in_int_block = True
        elif not is_int and in_int_block:
            lines.append("    MARKER                 'MARKER'                 'INTEND'")
            in_int_block = False

        entries = []
        if vname in obj_coeffs:
            entries.append(("COST", obj_coeffs[vname]))
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
        lines.append("    MARKER                 'MARKER'                 'INTEND'")

    lines.append("RHS")
    # In standard MPS, the RHS entry on the objective row is the negated objective constant
    if obj_const != 0.0:
        lines.append(f"    RHS       COST      {-obj_const:<12g}")
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
    obj_coeffs, obj_const, constraints = validate_and_extract(spec)
    write_mps(spec, obj_coeffs, obj_const, constraints, output_path, problem_name=problem_name)
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