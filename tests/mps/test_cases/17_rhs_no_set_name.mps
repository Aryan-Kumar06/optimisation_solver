NAME          RHS_NOSETNAME
* RHS and RANGES records with the set name OMITTED -- an even field count of
* bare (row, value) pairs. Netlib's blend is written this way, and consuming
* field 1 unconditionally made the whole file unreadable: the reader took a row
* name for the set name and then failed on "RHS entry for row '5.25' has no
* value".
* Expected: C1 [-inf,10], C2 [5,inf] with a range of 4 giving [5,9].
ROWS
 N  OBJ
 L  C1
 G  C2
COLUMNS
    X         OBJ       1.0   C1        1.0
    X         C2        1.0
RHS
    C1        10.0  C2        5.0
RANGES
    C2        4.0
ENDATA
