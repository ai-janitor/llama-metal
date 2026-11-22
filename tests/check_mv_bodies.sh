#!/bin/bash
# Host check of tests/mv-bodies.tsv against the Metal sources: every export whose name starts with kernel_mul_mv or
# kernel_norm_mul_mv has exactly one row, every row has its export, and a row with case none has a reason.
# Exports are the host_name("...") lines, the plain kernel functions that are not templates, and the names an entry
# macro makes from its first argument. Usage: check_mv_bodies.sh <table> <metal source dir>. Exit 1 on any finding.
table=$1; dir=$2
[ -f "$table" ] && [ -d "$dir" ] || { echo "usage: check_mv_bodies.sh <table> <metal source dir>"; exit 2; }
exports=$(find "$dir" -name '*.metal' -print0 | xargs -0 perl -ne '
    BEGIN { $re = qr/kernel_(?:norm_)?mul_mv\w*/; }
    if (/^\s*#\s*define\s+(\w+)\(\s*(\w+)/) { $mac = $1; $arg = $2; }
    if (defined $mac && /kernel\s+void\s+(\w*)##\Q$arg\E##(\w*)\s*\(/) { $pre{$mac} = $1; $suf{$mac} = $2; }
    if (defined $mac && /kernel\s+void\s+\Q$arg\E\s*\(/) { $pre{$mac} = ""; $suf{$mac} = ""; }
    $mac = undef unless /\\\s*$/;
    while (/host_name\("($re)"\)/g) { print "$1\n"; }
    if (/^\s*kernel\s+void\s+($re)\s*\(/ && $prev !~ /^\s*template\b/) { print "$1\n"; }
    if (/^\s*(\w+)\s*\(\s*(\w+)\s*[,)]/ && exists $pre{$1}) { $n = $pre{$1} . $2 . $suf{$1}; print "$n\n" if $n =~ /^$re$/; }
    $prev = $_ if /\S/;
    ' | sort -u)
rows=$(grep -v '^#' "$table" | tail -n +2 | cut -f1 | sort)
bad=0
for e in $(comm -23 <(echo "$exports") <(echo "$rows" | sort -u)); do echo "UNLISTED-EXPORT $e"; bad=$((bad + 1)); done
for e in $(comm -13 <(echo "$exports") <(echo "$rows" | sort -u)); do echo "STALE-ROW $e"; bad=$((bad + 1)); done
for e in $(echo "$rows" | uniq -d); do echo "DUPLICATE-ROW $e"; bad=$((bad + 1)); done
for e in $(grep -v '^#' "$table" | tail -n +2 | awk -F '\t' '$5 == "none" && $12 == "" { print $1 }'); do echo "NO-REASON $e"; bad=$((bad + 1)); done
echo "mv bodies: $(echo "$exports" | grep -c .) exports, $(echo "$rows" | grep -c .) rows, $bad findings"
[ $bad -eq 0 ]
