# Netlib LP instances

Source: https://netlib.org/lp/data/

Netlib does not distribute plain MPS. Each instance is stored in a compressed
form that must be expanded with the `emps` decompressor published alongside it
(https://netlib.org/lp/data/emps.c). A pipeline that downloads these files and
feeds them straight to a reader sees line-noise, not a model.

    cc -o emps emps.c
    ./emps afiro > afiro.mps

Reference optimal values are from the table in https://netlib.org/lp/data/readme
and are external claims used for corroboration only -- never as an optimality
proof. See benchmarks/README.md.

| instance | rows | cols | nonzeros | published optimum   |
|----------|------|------|----------|---------------------|
| afiro    | 28   | 32   | 88       | -4.6475314286E+02   |
