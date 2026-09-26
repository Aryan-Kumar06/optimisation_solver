NAME          FREE_FIXED
* min x + y + z  s.t.  x + z >= -4
* x FR (free), y FX at 3, z MI with UP 5.
* x + z is driven to -4; y is pinned at 3. Optimum obj = -1.
ROWS
 N  OBJ
 G  C1
COLUMNS
    X         OBJ       1.0   C1        1.0
    Y         OBJ       1.0
    Z         OBJ       1.0   C1        1.0
RHS
    RHS1      C1        -4.0
BOUNDS
 FR BND1      X
 FX BND1      Y         3.0
 MI BND1      Z
 UP BND1      Z         5.0
ENDATA
