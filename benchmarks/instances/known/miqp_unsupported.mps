NAME          MIQP_UNSUP
* Quadratic objective AND integrality: no engine does branch-and-cut over a
* quadratic relaxation, so the dispatcher must refuse rather than answer.
ROWS
 N  OBJ
 G  C1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         C1        1.0
    Y         C1        1.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS1      C1        2.0
QUADOBJ
    X         X         2.0
    Y         Y         2.0
ENDATA
