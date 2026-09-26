NAME          MAXOFF_LP
* max 2x + 3y + 100  s.t.  x + y <= 4, x + 3y <= 6, x,y >= 0
*
* The RHS on the objective row is the NEGATED constant, so -100.0 means +100.
* Vertices: (0,0)->100, (4,0)->108, (3,1)->109, (0,2)->106. Optimum 109 at
* (3,1), where both rows bind. Shadow prices: A'y = c gives y1 = 1.5, y2 = 0.5,
* and 4*1.5 + 6*0.5 + 100 = 109 reproduces the objective.
*
* Two-term rows, so presolve cannot reduce this to a bound walk and the sense
* flip and the objective constant are both carried through a real solve.
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  C1
 L  C2
COLUMNS
    X         OBJ       2.0   C1        1.0
    X         C2        1.0
    Y         OBJ       3.0   C1        1.0
    Y         C2        3.0
RHS
    RHS1      C1        4.0   C2        6.0
    RHS1      OBJ       -100.0
ENDATA
