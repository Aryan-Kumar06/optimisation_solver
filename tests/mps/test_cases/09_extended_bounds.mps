NAME          EXTBOUNDS
ROWS
 N  OBJ
 L  C1
COLUMNS
    X_MI      OBJ       1.0   C1        1.0
    X_PL      OBJ       1.0   C1        1.0
    X_BV      OBJ       1.0   C1        1.0
    X_LI      OBJ       1.0   C1        1.0
    X_UI      OBJ       1.0   C1        1.0
    X_NEGUP   OBJ       1.0   C1        1.0
RHS
    RHS1      C1        10.0
BOUNDS
 MI BND1      X_MI
 UP BND1      X_MI      4.0
 PL BND1      X_PL
 BV BND1      X_BV
 LI BND1      X_LI      2.0
 UI BND1      X_UI      9.0
 UP BND1      X_NEGUP   -3.0
ENDATA
