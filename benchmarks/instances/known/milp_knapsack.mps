NAME          KNAPSACK
* max 5a + 4b + 3c over binaries, three cover rows.
* Exhaustive check: (1,1,0) is the only feasible point scoring 9.
OBJSENSE
    MAX
ROWS
 N  OBJ
 L  C1
 L  C2
 L  C3
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    A         OBJ       5.0   C1        2.0
    A         C2        4.0   C3        3.0
    B         OBJ       4.0   C1        3.0
    B         C2        1.0   C3        4.0
    C         OBJ       3.0   C1        1.0
    C         C2        2.0   C3        2.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS1      C1        5.0   C2        11.0
    RHS1      C3        8.0
BOUNDS
 BV BND1      A
 BV BND1      B
 BV BND1      C
ENDATA
