# Puts rows of [matvec_shapes] into a copy of data/matmul.tsv, as its header says a sweep's rows go in:
#   awk -F'\t' -v rows=ROWS.tsv -f matvec_rows_put.awk matmul.tsv > copy.tsv
# ROWS.tsv: rows of one card, in the order the sweep printed them. A row of the table with the same card, kind,
# weight, k, m, cols and prep is replaced; the new rows stand together where the card's first row stood, or at the
# end of the section when the card has none.
function key(line,    f) { split(line, f, "\t"); return f[1] "\t" f[4] "\t" f[5] "\t" f[6] "\t" f[7] "\t" f[8] "\t" f[9] }
function put(    i) { if (!placed) { for (i = 1; i <= n; i++) print fresh[i]; placed = 1 } }
BEGIN {
    while ((getline line < rows) > 0) { if (line == "") continue; fresh[++n] = line; taken[key(line)] = 1; split(line, c, "\t"); card = c[1] }
    if (n == 0) { print "matvec_rows_put.awk: no row in " rows > "/dev/stderr"; exit 2 }
}
/^\[/ { if (in_s) put(); in_s = ($0 == "[matvec_shapes]"); print; next }
in_s && $0 == "" { put(); in_s = 0; print; next }
in_s && $1 == card { put() }
in_s && (key($0) in taken) { next }
{ print }
END { if (in_s) put(); if (n > 0 && !placed) { print "matvec_rows_put.awk: the table has no section [matvec_shapes]" > "/dev/stderr"; exit 2 } }
