NAME          MAX_QP
* max -x^2 - y^2 + 4x + 4y  s.t.  x + y <= 3
* QUADOBJ states 0.5*Q*x^2, so Q = -2 on each diagonal gives -x^2 - y^2.
* The equivalent minimisation is convex, which is what the ADMM engine needs.
* Unconstrained optimum is x=y=2 with x+y=4, so the row binds at x=y=1.5:
*   obj = -2.25 - 2.25 + 6 + 6 = 7.5
* Shadow price: with x=y=u/2, f(u) = -u^2/2 + 4u, so df/du = -u + 4 = +1 at
* u=3. POSITIVE, because relaxing a binding <= row of a maximisation improves
* the objective. A two-term row so presolve cannot turn it into a bound.
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  C1
COLUMNS
    X         OBJ       4.0   C1        1.0
    Y         OBJ       4.0   C1        1.0
RHS
    RHS1      C1        3.0
QUADOBJ
    X         X         -2.0
    Y         Y         -2.0
ENDATA
