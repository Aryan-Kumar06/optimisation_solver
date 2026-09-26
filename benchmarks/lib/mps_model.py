"""An independent reader for MPS/QPS, written from the format definition.

This exists to be a SECOND opinion, not a better one. The checker must not
verify a solution against the same parse that produced it: if the C++ reader
misreads a RANGES record, a checker sharing that reader would confirm the wrong
answer as correct, and the whole pipeline would agree with itself.

So this file shares no code with src/mps/mps_reader.cpp. It also differs
structurally on purpose -- rows are dicts keyed by column index, bounds are
resolved in a separate pass, and the quadratic form is normalised at parse time
-- so that an off-by-one or an ordering assumption is unlikely to appear in
both.

HONEST LIMIT: both readers were written by the same author, so correlated
blind spots are possible. The independent check that does NOT share an author
is agreement with HiGHS on the objective value, which is fed from THIS model.
Treat parse agreement as necessary, not sufficient.

Model convention, matching model::Model so the two are directly comparable:

    f(x) = offset + sum_j c_j x_j + sum_(i<=j) q_ij x_i x_j

where q_ij is the DIRECT coefficient of x_i*x_j with no implicit 1/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = math.inf


class MpsError(Exception):
    pass


@dataclass
class Model:
    name: str = ""
    sense: str = "min"           # "min" or "max"
    offset: float = 0.0

    obj_linear: dict = field(default_factory=dict)   # col -> coefficient
    obj_quad: dict = field(default_factory=dict)     # (i, j) with i <= j -> coefficient

    var_names: list = field(default_factory=list)
    var_lower: list = field(default_factory=list)
    var_upper: list = field(default_factory=list)
    var_type: list = field(default_factory=list)     # "C", "I", "B"

    row_names: list = field(default_factory=list)
    row_lower: list = field(default_factory=list)
    row_upper: list = field(default_factory=list)
    rows: list = field(default_factory=list)         # list of {col: coefficient}

    warnings: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.var_names)

    @property
    def m(self) -> int:
        return len(self.row_names)

    def is_integer_model(self) -> bool:
        return any(t != "C" for t in self.var_type)

    def is_quadratic(self) -> bool:
        return bool(self.obj_quad)

    def problem_class(self) -> str:
        q, i = self.is_quadratic(), self.is_integer_model()
        return ("MIQP" if i else "QP") if q else ("MILP" if i else "LP")

    def row_activity(self, x) -> list:
        return [sum(coefficient * x[column] for column, coefficient in row.items())
                for row in self.rows]

    def objective(self, x) -> float:
        """Objective of the ORIGINAL model, including the constant."""
        total = self.offset
        for column, coefficient in self.obj_linear.items():
            total += coefficient * x[column]
        for (i, j), coefficient in self.obj_quad.items():
            total += coefficient * x[i] * x[j]
        return total

    def gradient(self, x) -> list:
        """d f / d x, following the direct-coefficient convention.

        A diagonal term q_kk contributes 2*q_kk*x_k; an off-diagonal q_ij
        contributes q_ij*x_j to component i and q_ij*x_i to component j.
        """
        grad = [0.0] * self.n
        for column, coefficient in self.obj_linear.items():
            grad[column] += coefficient
        for (i, j), coefficient in self.obj_quad.items():
            if i == j:
                grad[i] += 2.0 * coefficient * x[i]
            else:
                grad[i] += coefficient * x[j]
                grad[j] += coefficient * x[i]
        return grad

    def to_minimization(self) -> "Model":
        """An equivalent minimisation model, so checks have one sign convention.

        Duals and reduced costs of a maximisation must be negated alongside,
        because d(-f)/db = -df/db. The checker does exactly that.
        """
        if self.sense == "min":
            return self
        flipped = Model(
            name=self.name, sense="min", offset=-self.offset,
            obj_linear={k: -v for k, v in self.obj_linear.items()},
            obj_quad={k: -v for k, v in self.obj_quad.items()},
            var_names=self.var_names, var_lower=self.var_lower,
            var_upper=self.var_upper, var_type=self.var_type,
            row_names=self.row_names, row_lower=self.row_lower,
            row_upper=self.row_upper, rows=self.rows, warnings=self.warnings,
        )
        return flipped


_BOUND_NEEDS_VALUE = {"LO", "UP", "FX", "LI", "UI"}
_KNOWN_SECTIONS = {
    "NAME", "OBJSENSE", "OBJSENS", "ROWS", "COLUMNS", "RHS", "RANGES",
    "BOUNDS", "QUADOBJ", "QUADS", "QMATRIX", "QUADOBJ2", "ENDATA",
}
_REJECTED_SECTIONS = {"SOS", "INDICATORS", "QCMATRIX", "QCROWS"}


def read_mps(path: str) -> Model:
    model = Model()

    # Parsed into name-keyed staging first, then indexed. Keeping the two
    # stages apart is what lets RANGES be applied after RHS regardless of the
    # order they appear in.
    obj_row = None
    free_rows = set()
    row_sense = {}          # name -> "L" | "G" | "E"
    row_rhs = {}            # name -> float
    row_range = {}          # name -> float
    row_terms = {}          # name -> {var_name: coefficient}
    row_order = []

    col_index = {}
    col_order = []
    col_lower = {}
    col_upper = {}
    col_type = {}
    col_lower_explicit = set()

    obj_terms = {}
    quad_terms = {}
    objective_constant = 0.0

    # A file may define several RHS or RANGES vectors; they are ALTERNATIVES,
    # not accumulations, and only the first belongs to the problem being solved.
    # Taking the last value seen (which this reader did) silently solves a
    # different instance -- caught against tests/mps/test_cases/17_multiple_rhs
    # and 18_multiple_ranges, where the alternative vectors are all 999.
    selected_rhs_vector = None
    selected_ranges_vector = None

    section = None
    integer_mode = False

    def touch_column(name):
        if name not in col_index:
            col_index[name] = len(col_order)
            col_order.append(name)
            col_lower[name] = 0.0
            col_upper[name] = INF
            # A general integer defaults to [0, +inf). The older [0, 1] default
            # silently converts general integers into binaries.
            col_type[name] = "I" if integer_mode else "C"
        return col_index[name]

    with open(path, "r", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line or line[0] == "*":
                continue

            if line[0] not in (" ", "\t"):
                header = line.split()[0].upper()
                if header in _REJECTED_SECTIONS:
                    raise MpsError(
                        f"line {line_number}: section {header} changes the problem "
                        f"and is not representable")
                if header not in _KNOWN_SECTIONS:
                    raise MpsError(
                        f"line {line_number}: unrecognised section '{header}'")
                section = header
                fields = line.split()
                if header == "NAME" and len(fields) > 1:
                    model.name = fields[1]
                if header in ("OBJSENSE", "OBJSENS") and len(fields) > 1:
                    model.sense = _parse_sense(fields[1], line_number)
                if header == "ENDATA":
                    break
                continue

            fields = line.split()
            if not fields:
                continue

            if section in ("OBJSENSE", "OBJSENS"):
                model.sense = _parse_sense(fields[0], line_number)

            elif section == "ROWS":
                if len(fields) < 2:
                    raise MpsError(f"line {line_number}: malformed ROWS record")
                sense, name = fields[0].upper(), fields[1]
                if sense == "N":
                    if obj_row is None:
                        obj_row = name
                    else:
                        free_rows.add(name)
                        model.warnings.append(f"free row '{name}' ignored")
                elif sense in ("L", "G", "E"):
                    if name in row_sense:
                        raise MpsError(f"line {line_number}: duplicate row '{name}'")
                    row_sense[name] = sense
                    row_rhs[name] = 0.0
                    row_terms[name] = {}
                    row_order.append(name)
                else:
                    raise MpsError(f"line {line_number}: unknown row sense '{sense}'")

            elif section == "COLUMNS":
                # An integer marker is
                #     <name>  'MARKER'  'INTORG'|'INTEND'
                # identified by FIELDS 2 and 3, never by searching the line for
                # the substring "MARKER". Substring matching swallowed real
                # columns whose NAMES contain the word -- X_MARKER_INTORG,
                # INTEGER_MARKER_INTEND, a column simply called MARKER, and a
                # coefficient on a ROW called MARKER all appear in
                # tests/mps/test_cases/19_marker_names, and this reader lost
                # three of the five columns to them.
                if len(fields) >= 3 and _unquote(fields[1]).upper() == "MARKER":
                    keyword = _unquote(fields[2]).upper()
                    if keyword == "INTORG":
                        integer_mode = True
                        continue
                    if keyword == "INTEND":
                        integer_mode = False
                        continue
                var_name = fields[0]
                touch_column(var_name)
                rest = fields[1:]
                if len(rest) % 2 != 0:
                    raise MpsError(
                        f"line {line_number}: COLUMNS record has an unpaired entry")
                for k in range(0, len(rest), 2):
                    target, value = rest[k], _to_float(rest[k + 1], line_number)
                    if target == obj_row:
                        obj_terms[var_name] = obj_terms.get(var_name, 0.0) + value
                    elif target in row_terms:
                        row_terms[target][var_name] = \
                            row_terms[target].get(var_name, 0.0) + value
                    elif target not in free_rows:
                        model.warnings.append(
                            f"COLUMNS entry names unknown row '{target}'")

            elif section == "RHS":
                # The set name is optional: records are (row, value) pairs, so an
                # odd field count carries a name and an even count does not.
                named = (len(fields) % 2 == 1)
                if named:
                    if selected_rhs_vector is None:
                        selected_rhs_vector = fields[0]
                    elif fields[0] != selected_rhs_vector:
                        model.warnings.append(
                            f"RHS vector '{fields[0]}' ignored; using "
                            f"'{selected_rhs_vector}'")
                        continue
                rest = fields[1:] if named else fields
                for k in range(0, len(rest) - 1, 2):
                    target, value = rest[k], _to_float(rest[k + 1], line_number)
                    if target == obj_row:
                        # MPS defines this as the NEGATED objective constant.
                        objective_constant = -value
                    elif target in row_rhs:
                        row_rhs[target] = value
                    elif target not in free_rows:
                        model.warnings.append(
                            f"RHS entry names unknown row '{target}'")

            elif section == "RANGES":
                named = (len(fields) % 2 == 1)
                if named:
                    if selected_ranges_vector is None:
                        selected_ranges_vector = fields[0]
                    elif fields[0] != selected_ranges_vector:
                        model.warnings.append(
                            f"RANGES vector '{fields[0]}' ignored; using "
                            f"'{selected_ranges_vector}'")
                        continue
                rest = fields[1:] if named else fields
                for k in range(0, len(rest) - 1, 2):
                    target, value = rest[k], _to_float(rest[k + 1], line_number)
                    if target in row_rhs:
                        row_range[target] = value
                    else:
                        model.warnings.append(
                            f"RANGES entry names unknown row '{target}'")

            elif section == "BOUNDS":
                kind = fields[0].upper().strip("'\"")
                if len(fields) < 3:
                    raise MpsError(f"line {line_number}: malformed BOUNDS record")
                var_name = fields[2]
                touch_column(var_name)
                value = None
                if kind in _BOUND_NEEDS_VALUE:
                    if len(fields) < 4:
                        raise MpsError(
                            f"line {line_number}: bound {kind} requires a value")
                    value = _to_float(fields[3], line_number)

                if kind == "LO":
                    col_lower[var_name] = value
                    col_lower_explicit.add(var_name)
                elif kind == "UP":
                    col_upper[var_name] = value
                    if value < 0.0 and var_name not in col_lower_explicit \
                            and col_lower[var_name] == 0.0:
                        col_lower[var_name] = -INF
                        model.warnings.append(
                            f"negative UP on '{var_name}' opened the lower bound")
                elif kind == "FX":
                    col_lower[var_name] = value
                    col_upper[var_name] = value
                    col_lower_explicit.add(var_name)
                elif kind == "FR":
                    col_lower[var_name] = -INF
                    col_upper[var_name] = INF
                    col_lower_explicit.add(var_name)
                elif kind == "MI":
                    col_lower[var_name] = -INF
                    col_lower_explicit.add(var_name)
                elif kind == "PL":
                    col_upper[var_name] = INF
                elif kind == "BV":
                    col_lower[var_name] = 0.0
                    col_upper[var_name] = 1.0
                    col_type[var_name] = "B"
                    col_lower_explicit.add(var_name)
                elif kind == "LI":
                    col_lower[var_name] = value
                    col_lower_explicit.add(var_name)
                    if col_type[var_name] == "C":
                        col_type[var_name] = "I"
                elif kind == "UI":
                    col_upper[var_name] = value
                    if col_type[var_name] == "C":
                        col_type[var_name] = "I"
                else:
                    raise MpsError(
                        f"line {line_number}: unknown bound type '{kind}'")

            elif section in ("QUADOBJ", "QUADS", "QMATRIX", "QUADOBJ2"):
                triangular = section in ("QUADOBJ", "QUADS")
                first = fields[0]
                rest = fields[1:]
                if len(rest) % 2 != 0:
                    raise MpsError(
                        f"line {line_number}: quadratic record has an unpaired entry")
                for k in range(0, len(rest), 2):
                    second = rest[k]
                    value = _to_float(rest[k + 1], line_number)
                    i = touch_column(first)
                    j = touch_column(second)
                    lo, hi = (i, j) if i <= j else (j, i)
                    # The file states 0.5*x'Qx; the model stores the direct
                    # coefficient. A triangular off-diagonal implies its mirror
                    # and so carries the whole Q_ij; a full-matrix one is listed
                    # twice and carries half each time.
                    if i == j:
                        contribution = 0.5 * value
                    else:
                        contribution = value if triangular else 0.5 * value
                    quad_terms[(lo, hi)] = quad_terms.get((lo, hi), 0.0) + contribution

    # ---- index everything -------------------------------------------------
    model.var_names = list(col_order)
    model.var_lower = [col_lower[c] for c in col_order]
    model.var_upper = [col_upper[c] for c in col_order]
    model.var_type = [col_type[c] for c in col_order]
    model.offset = objective_constant
    model.obj_linear = {col_index[c]: v for c, v in obj_terms.items() if v != 0.0}
    model.obj_quad = {k: v for k, v in quad_terms.items() if v != 0.0}

    for name in row_order:
        model.row_names.append(name)
        model.rows.append({col_index[c]: v for c, v in row_terms[name].items()})
        lower, upper = _resolve_row_bounds(
            row_sense[name], row_rhs[name], row_range.get(name))
        model.row_lower.append(lower)
        model.row_upper.append(upper)

    return model


def _resolve_row_bounds(sense: str, rhs: float, rng):
    """Sense + RHS + RANGES -> a bound pair.

    The width is |R| for every sense. Only an equality row takes its SIDE from
    the sign of R, which is the part of the spec most often implemented wrong.
    """
    if rng is None:
        if sense == "L":
            return -INF, rhs
        if sense == "G":
            return rhs, INF
        return rhs, rhs

    width = abs(rng)
    if sense == "L":
        return rhs - width, rhs
    if sense == "G":
        return rhs, rhs + width
    return (rhs, rhs + rng) if rng >= 0.0 else (rhs + rng, rhs)


def _unquote(token: str) -> str:
    """Marker keywords are quoted; the quoting style varies by writer."""
    return token.strip("'\"")


def _parse_sense(token: str, line_number: int) -> str:
    upper = token.upper()
    if upper in ("MAX", "MAXIMIZE"):
        return "max"
    if upper in ("MIN", "MINIMIZE"):
        return "min"
    raise MpsError(f"line {line_number}: unknown OBJSENSE '{token}'")


def _to_float(token: str, line_number: int) -> float:
    try:
        return float(token)
    except ValueError as error:
        raise MpsError(f"line {line_number}: '{token}' is not a number") from error
