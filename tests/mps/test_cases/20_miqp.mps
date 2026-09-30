NAME          MIQPPIPE
ROWS
 N  OBJ
 E  EQ
 G  GE
 L  LE
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         OBJ      -3.0   EQ        1.0
    X         GE        1.0
    MARKER                 'MARKER'                 'INTEND'
    Y         OBJ      -1.0   EQ       -1.0
    Y         GE        1.0   LE        1.0
RHS
    RHS1      EQ        1.0   GE        2.0
    RHS1      LE        1.0
BOUNDS
 LO BND1      X         0.0
 UP BND1      X         3.0
 LO BND1      Y         0.0
 UP BND1      Y         2.0
QUADOBJ
    X         X         2.0
    Y         Y         2.0
ENDATA
