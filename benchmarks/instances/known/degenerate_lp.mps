NAME          DEGEN_LP
* min -x - y  s.t.  x + y <= 4, x + 2y <= 6, 2x + y <= 6, x,y >= 0
*
* Optimum (2,2), obj = -4. All THREE rows are tight there, on two variables:
* the optimal basis is primal degenerate and the duals are not unique.
*
* Every row carries two terms so presolve cannot rewrite any of them as a
* bound. The earlier version of this instance used singleton rows, which
* presolve turned into bounds until nothing was left -- it routed to the
* trivial bound-walk and never reached the simplex at all, so it tested
* presolve rather than degeneracy.
ROWS
 N  OBJ
 L  C1
 L  C2
 L  C3
COLUMNS
    X         OBJ       -1.0  C1        1.0
    X         C2        1.0   C3        2.0
    Y         OBJ       -1.0  C1        1.0
    Y         C2        2.0   C3        1.0
RHS
    RHS1      C1        4.0   C2        6.0
    RHS1      C3        6.0
ENDATA
