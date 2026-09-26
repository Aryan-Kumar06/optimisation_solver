NAME          OPT_LP
* max 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18
* Optimum x=2 y=6 obj=36; duals [0, 1.5, 1] (matches HiGHS marginals).
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  C1
 L  C2
 L  C3
COLUMNS
    X         OBJ       3.0   C1        1.0
    X         C3        3.0
    Y         OBJ       5.0   C2        2.0
    Y         C3        2.0
RHS
    RHS1      C1        4.0   C2        12.0
    RHS1      C3        18.0
ENDATA
