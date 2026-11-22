#!/bin/bash
# Source check of the one-column matvec bodies: every weight row address in ggml/src/ggml-metal/matmul/*.metal is
# formed from the bounded row (the word rb or mv_tail_row), and every line that names the bounded row or the
# whole-rows selection (rows_whole, mv_rows_whole) is a reviewed pin. Each allow and pin row is found exactly count times.
# Usage: check_mv_row_bound.sh <table tests/mv-row-bound-allow.tsv> <matmul dir>. Table rows: file, kind, line, why, count
# (see the table's header). Exit 1 on any finding.
table=$1; dir=$2
[ -f "$table" ] && [ -d "$dir" ] || { echo "usage: check_mv_row_bound.sh <table> <matmul dir>"; exit 2; }
perl -e '
    my ($table, $dir) = @ARGV;
    my (%allow, %pin, %want, %found, %whole, %used);
    my $bad = 0; my $files = 0;
    open(my $t, "<", $table) or die "no table\n";
    while (<$t>) {
        chomp; next if /^#/ || /^\s*$/;
        my ($f, $k, $l, $why, $count) = split /\t/;
        next if $f eq "file" && $k eq "kind";
        if ($k eq "file") { $whole{$f} = 1; next; }
        my $key = "$f\t$l";
        if (($k ne "allow" && $k ne "pin") || !defined $count || $count !~ /^[1-9][0-9]*$/) { print "BAD-ROW $f $k $l\n"; $bad++; next; }
        if ($k eq "allow") { $allow{$key} = 1; } else { $pin{$key} = 1; }
        $want{$key} = $count;
    }
    close($t);
    opendir(my $d, $dir) or die "no dir\n";
    for my $name (sort grep { /\.metal$/ } readdir($d)) {
        my $f = "matmul/$name";
        $files++;
        if ($whole{$f}) { $used{"$f\t*"} = 1; next; }
        open(my $h, "<", "$dir/$name") or die;
        my $n = 0;
        while (my $line = <$h>) {
            $n++; chomp $line;
            (my $s = $line) =~ s/^\s+|\s+$//g;
            next if $s eq "" || $s =~ m{^//};
            my $key = "$f\t$s";
            $found{$key}++ if exists $want{$key};
            if (($s =~ /\brows_whole\b/ || $s =~ /mv_rows_whole\(/) && $s !~ /^static inline bool mv_rows_whole\(/) {
                if (!$pin{$key}) { print "PIN-OTHER $f:$n\n"; $bad++; }
                next;
            }
            my $names_rb = ($s =~ /\brb\b/) || ($s =~ /mv_tail_row\(/);
            if ($names_rb && $s !~ /^static inline int mv_tail_row\(/) {
                if (!$pin{$key}) { print "PIN-OTHER $f:$n\n"; $bad++; }
                next if ($s =~ /mv_tail_row\(/) || $s !~ /\b(?:r0 \+ row|first_row \+ row|first_row \+ ltid)\b/;
            }
            my $stride  = $s =~ /nb01|ns01/;
            my $logical = $s =~ /r0 \+ row|first_row \+ row|first_row \+ ltid/;
            my $unbound = ($stride && !$names_rb) || ($logical && $s !~ /mv_tail_row\(/);
            next unless $unbound;
            if (!$allow{$key}) { print "UNBOUND $f:$n\n"; $bad++; }
        }
        close($h);
    }
    for my $k (sort keys %want) {
        my ($f, $l) = split /\t/, $k, 2; my $k2 = $found{$k} // 0;
        if ($k2 == 0) { print "STALE-ALLOW $f: $l\n"; $bad++; }
        elsif ($k2 != $want{$k}) { print "ALLOW-COUNT $f · $l · expected $want{$k} · found $k2\n"; $bad++; }
    }
    for my $f (sort keys %whole) { next if $used{"$f\t*"}; print "STALE-ALLOW $f: *\n"; $bad++; }
    print "mv row bound: $files files, $bad findings\n";
    exit($bad ? 1 : 0);
' "$table" "$dir"
