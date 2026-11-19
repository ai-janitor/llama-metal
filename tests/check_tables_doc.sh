#!/bin/sh
# Checks "## The tables" of ggml/src/ggml-metal/README.md: every .tsv of the folder is listed there, and every
# `file:line` (`name`) it cites still holds that name on that line. Checks every .tsv itself: each column of its header
# row has a comment line "#   <column>: " in the same file.
# usage: check_tables_doc.sh <ggml-metal dir> [extra .tsv name that must be reported missing, or -] [README to read instead]
dir="$1"
extra="$2"
readme="${3:-$dir/README.md}"
if [ "$extra" = "-" ]; then extra=""; fi
sec=$(awk '/^## The tables$/ { on = 1; next } /^## / { on = 0 } on' "$readme")
if [ -z "$sec" ]; then
    echo "FAIL tables doc: README.md has no '## The tables' section"
    exit 1
fi
bad=0
n=0
for f in $(cd "$dir" && find . -name '*.tsv' | sed 's|^\./||' | sort) $extra; do
    n=$((n + 1))
    if ! printf '%s\n' "$sec" | grep -qF "| \`$f\` |"; then
        echo "FAIL tables doc: $f has no row in '## The tables'"
        bad=$((bad + 1))
    fi
done
cols=0
for f in $(cd "$dir" && find . -name '*.tsv' | sed 's|^\./||' | sort); do
    for c in $(grep -v '^#' "$dir/$f" | head -1 | tr '\t' ' '); do
        cols=$((cols + 1))
        if ! grep -q "^#   $c: " "$dir/$f"; then
            echo "FAIL tables doc: $f has no comment for column $c"
            bad=$((bad + 1))
        fi
    done
done
refs=0
for ref in $(printf '%s\n' "$sec" | grep -oE '`[a-z0-9/._-]+:[0-9]+` \(`[A-Za-z0-9_]+`\)' | sed 's/` (`/:/' | tr -d '`)'); do
    file=$(echo "$ref" | cut -d: -f1)
    line=$(echo "$ref" | cut -d: -f2)
    name=$(echo "$ref" | cut -d: -f3)
    refs=$((refs + 1))
    if ! sed -n "${line}p" "$dir/$file" 2>/dev/null | grep -qF "$name"; then
        echo "FAIL tables doc: $file:$line does not hold $name"
        bad=$((bad + 1))
    fi
done
echo "tables doc: $n tables, $cols columns, $refs cited lines, $bad failures"
[ "$bad" -eq 0 ]
