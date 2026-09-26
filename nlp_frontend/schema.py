"""
Strict schema validation for the LLM-extracted optimization problem spec.

Fails closed: any malformed, missing, inconsistent, or out-of-range field
raises SchemaValidationError with a specific message instead of silently
coercing, defaulting, or truncating.

Addresses PR #9 review items:
  3. Strict schema validation of the LLM's JSON output.
  4. No silent lower_bound=0 default -- bounds are only ever None
     (meaning "not specified"), a real number, or rejected outright.
  5. Variable/constraint name validation before anything touches
     sympy or the MPS writer.
"""

import re

VALID_VAR_TYPES = {"continuous", "integer", "binary"}
VALID_SENSES = {"maximize", "minimize"}
VALID_OPERATORS = {"<=", ">=", "="}

# Fixed-format MPS caps entity names at 8 characters. We enforce the
# stricter fixed-format limit so generated files work under either
# fixed- or free-format readers.
MAX_MPS_NAME_LEN = 8
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class SchemaValidationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise SchemaValidationError(message)


def validate_name(name, context):
    _require(isinstance(name, str) and name, f"{context}: name must be a non-empty string")
    _require(
        bool(NAME_RE.match(name)),
        f"{context}: '{name}' is not a valid identifier "
        f"(must start with a letter, then only letters/digits/underscore)",
    )
    _require(
        len(name) <= MAX_MPS_NAME_LEN,
        f"{context}: '{name}' exceeds {MAX_MPS_NAME_LEN} characters (fixed-format MPS name limit)",
    )


def validate_bound(value, context):
    """None means 'not specified' -- that is a valid, explicit state.
    Anything else must be a real number, never a stand-in default."""
    if value is None:
        return
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context}: bound must be numeric or null, got {value!r}",
    )


def validate_spec(spec: dict) -> dict:
    """
    Validates the raw dict parsed from the LLM's JSON output.
    Returns the same dict on success; raises SchemaValidationError
    on the first violation found.
    """
    _require(isinstance(spec, dict), "Top-level spec must be a JSON object")

    for key in ("variables", "objective", "constraints"):
        _require(key in spec, f"Spec is missing required key '{key}'")

    # ---- variables ----
    variables = spec["variables"]
    _require(isinstance(variables, list) and len(variables) > 0, "'variables' must be a non-empty list")

    seen_names = set()
    for i, v in enumerate(variables):
        ctx = f"variables[{i}]"
        _require(isinstance(v, dict), f"{ctx}: must be an object")
        _require("name" in v, f"{ctx}: missing 'name'")
        validate_name(v["name"], ctx)
        _require(v["name"] not in seen_names, f"{ctx}: duplicate variable name '{v['name']}'")
        seen_names.add(v["name"])

        _require(
            "type" in v and v["type"] in VALID_VAR_TYPES,
            f"{ctx}: 'type' must be one of {sorted(VALID_VAR_TYPES)}, got {v.get('type')!r}",
        )

        lb = v.get("lower_bound")
        ub = v.get("upper_bound")
        validate_bound(lb, f"{ctx}.lower_bound")
        validate_bound(ub, f"{ctx}.upper_bound")
        if lb is not None and ub is not None:
            _require(lb <= ub, f"{ctx}: lower_bound ({lb}) exceeds upper_bound ({ub})")

        if v["type"] == "binary":
            if lb is not None:
                _require(lb in (0, 1), f"{ctx}: binary lower_bound must be 0 or 1, got {lb}")
            if ub is not None:
                _require(ub in (0, 1), f"{ctx}: binary upper_bound must be 0 or 1, got {ub}")

    var_names = seen_names

    # ---- objective ----
    obj = spec["objective"]
    _require(isinstance(obj, dict), "'objective' must be an object")
    _require(
        "sense" in obj and obj["sense"] in VALID_SENSES,
        f"objective.sense must be one of {sorted(VALID_SENSES)}, got {obj.get('sense')!r}",
    )
    _require(
        "expression" in obj and isinstance(obj["expression"], str) and obj["expression"].strip(),
        "objective.expression must be a non-empty string",
    )

    # ---- constraints ----
    constraints = spec["constraints"]
    _require(isinstance(constraints, list), "'constraints' must be a list")

    seen_cnames = set()
    for i, c in enumerate(constraints):
        ctx = f"constraints[{i}]"
        _require(isinstance(c, dict), f"{ctx}: must be an object")
        _require("name" in c, f"{ctx}: missing 'name'")
        validate_name(c["name"], ctx)
        _require(c["name"] not in seen_cnames, f"{ctx}: duplicate constraint name '{c['name']}'")
        _require(c["name"] not in var_names, f"{ctx}: constraint name '{c['name']}' collides with a variable name")
        seen_cnames.add(c["name"])

        _require(
            "expression" in c and isinstance(c["expression"], str) and c["expression"].strip(),
            f"{ctx}: expression must be a non-empty string",
        )
        _require(
            "operator" in c and c["operator"] in VALID_OPERATORS,
            f"{ctx}: operator must be one of {sorted(VALID_OPERATORS)}, got {c.get('operator')!r}",
        )
        _require(
            "rhs" in c and isinstance(c["rhs"], (int, float)) and not isinstance(c["rhs"], bool),
            f"{ctx}: rhs must be numeric, got {c.get('rhs')!r}",
        )

    return spec