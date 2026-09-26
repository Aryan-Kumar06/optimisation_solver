NAME          CONVEX_QP
* min x^2 + y^2 s.t. x + y >= 2, x,y >= 0.
* QUADOBJ states 0.5*x'Qx with Q = diag(2,2), giving x^2 + y^2.
* Optimum x=y=1, obj=2, row dual = 2.
ROWS
 N  OBJ
 G  C1
COLUMNS
    X         C1        1.0
    Y         C1        1.0
RHS
    RHS1      C1        2.0
QUADOBJ
    X         X         2.0
    Y         Y         2.0
ENDATA
