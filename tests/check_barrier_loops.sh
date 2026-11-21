#!/bin/sh
# Barriers inside loops and conditionals of the Metal kernels against tests/barrier-loops.tsv; rules L, H, C of the header of that file.
# usage: check_barrier_loops.sh scan DIR ROWS | check DIR ROWS FIXTURES (the self-test cases of FIXTURES, then scan DIR)
mode="$1"
if [ "$mode" = check ]; then
    dir="$2"; rows="$3"; fx="$4"; bad=0; n=0
    # case dir rows want(0|1) [line that must be printed]
    one() { n=$((n + 1)); out=$(sh "$0" scan "$fx/$1" "$fx/$2" 2>&1); rc=$?; got=0; [ $rc -ne 0 ] && got=1
        ok=1; [ "$got" = "$3" ] || ok=0
        if [ -n "$4" ]; then printf '%s\n' "$out" | grep -q -- "$4" || ok=0; fi
        if [ $ok = 1 ]; then echo "SELFTEST $1: exit $rc as expected"; else echo "SELFTEST $1: exit $rc, wanted $( [ "$3" = 0 ] && echo 0 || echo non-zero) $4 -- FAILED"; printf '%s\n' "$out"; bad=$((bad + 1)); fi; }
    one approved approved.tsv 0 "2 findings, 2 reviewed"
    one hash approved.tsv 1 "^BARRIER L"
    one stale approved.tsv 1 "^STALE-ROW"
    one cond empty.tsv 1 "^BARRIER C"
    one nobarrier empty.tsv 0 "0 findings"
    one else else.tsv 0 "1 findings, 1 reviewed"
    one elsepred else.tsv 1 "^BARRIER C"
    one head empty.tsv 1 "^BARRIER H"
    one unbraced empty.tsv 1 "^BARRIER H"
    one dowhile dowhile.tsv 0 "1 findings, 1 reviewed"
    one dowhilepred dowhile.tsv 1 "^BARRIER H"
    one commentdelims empty.tsv 1 "^BARRIER C"
    echo "self-test: $((n - bad)) of $n"
    [ $bad -eq 0 ] || exit 1
    exec sh "$0" scan "$dir" "$rows"
fi
[ "$mode" = scan ] && [ -d "$2" ] && [ -f "$3" ] || { echo "usage: check_barrier_loops.sh scan DIR ROWS | check DIR ROWS FIXTURES" >&2; exit 2; }
exec python3 - "$2" "$3" <<'EOF'
import hashlib, os, re, sys
root, rows_path = sys.argv[1], sys.argv[2]

def strip(s):  # one pass over the text: comments become blanks (string and char literals are skipped whole), then #if/#else/#endif lines
    out = list(s); i = 0; n = len(s)
    def blank(a, b):
        for k in range(a, b):
            if out[k] != '\n': out[k] = ' '
    while i < n:
        c = s[i]
        if c == '/' and s.startswith('//', i):
            j = s.find('\n', i); j = n if j < 0 else j; blank(i, j); i = j
        elif c == '/' and s.startswith('/*', i):
            j = s.find('*/', i + 2); j = n if j < 0 else j + 2; blank(i, j); i = j
        elif c in '"\'':
            j = i + 1
            while j < n and s[j] != c and s[j] != '\n':
                j += 2 if s[j] == '\\' else 1
            i = j + 1
        else:
            i += 1
    s = ''.join(out)
    return re.sub(r'(?m)^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*', lambda m: ' ' * len(m.group(0)), s)

def match(s, p, o, c):
    d = 0
    for i in range(p, len(s)):
        if s[i] == o: d += 1
        elif s[i] == c:
            d -= 1
            if d == 0: return i
    return -1

def skip_ws(s, p):
    while p < len(s) and s[p] in ' \t\r\n\\': p += 1
    return p

def stmt_end(s, p):
    d = 0
    for i in range(p, len(s)):
        ch = s[i]
        if ch in '({': d += 1
        elif ch in ')}': d -= 1
        elif ch == ';' and d == 0: return i
    return len(s) - 1

def body(s, b):  # (bs, be): a braced body from its brace; an unbraced statement from just before its first token, so it counts as inside
    return (b, match(s, b, '{', '}')) if s[b] == '{' else (b - 1, stmt_end(s, b))

def scan(rel, s):
    line = lambda p: s.count('\n', 0, p) + 1
    cons = []
    for m in re.finditer(r'\b(for|while|do|if|else|switch|FOR_UNROLL)\b', s):
        kw = m.group(1); q = skip_ws(s, m.end())
        if kw == 'else':
            if s.startswith('if', q) and not (q + 2 < len(s) and (s[q + 2].isalnum() or s[q + 2] == '_')):
                cons.append(dict(kw='else', start=m.start(), bs=q, be=-1, head='', elseif=True)); continue
            bs, be = body(s, q)
            cons.append(dict(kw='else', start=m.start(), bs=bs, be=be, head='', elseif=False)); continue
        if kw == 'do':
            bs, be = body(s, q); head = ''
            w = skip_ws(s, be + 1)  # the trailing while (predicate); is the loop's head and part of its block
            if s.startswith('while', w):
                pq = skip_ws(s, w + 5)
                if s[pq] == '(':
                    pe = match(s, pq, '(', ')'); head = s[pq:pe + 1]; be = stmt_end(s, pe + 1)
            cons.append(dict(kw='loop', start=m.start(), bs=bs, be=be, head=head, elseif=False)); continue
        if s[q] != '(': continue
        he = match(s, q, '(', ')'); head = s[q:he + 1]
        b = skip_ws(s, he + 1)
        if kw == 'while' and s[b] == ';': continue  # the tail of a do-while
        bs, be = body(s, b)
        cons.append(dict(kw=kw if kw in ('if', 'switch') else 'loop', start=m.start(), bs=bs, be=be, head=head, elseif=False))
    by_start = {c['start']: c for c in cons}
    def if_at(p):
        c = by_start.get(p)
        return c if c and c['kw'] == 'if' else None
    def chain_end(c):
        e = c['be']; q = skip_ws(s, e + 1)
        el = by_start.get(q)
        if not el or el['kw'] != 'else': return e
        return chain_end(if_at(el['bs'])) if el['elseif'] else el['be']
    for c in cons:
        if c['elseif']: c['be'] = chain_end(if_at(c['bs']))
    def chain_root(c):
        if c['kw'] == 'if':
            el = next((x for x in cons if x['kw'] == 'else' and x['elseif'] and x['bs'] == c['start']), None)
            return chain_root(el) if el else c
        if c['kw'] == 'else':
            pi = [x for x in cons if x['kw'] == 'if' and skip_ws(s, x['be'] + 1) == c['start']]
            return chain_root(pi[-1]) if pi else c
        return c
    bars = [m.start() for m in re.finditer(r'\b(simdgroup_barrier|threadgroup_barrier)\s*\(', s)]
    inside = lambda c, p: c['bs'] < p <= c['be']
    has_bar = lambda c: any(inside(c, p) for p in bars)
    out = []; roots = {}
    for p in bars:
        conds = [c for c in cons if c['kw'] in ('if', 'else', 'switch') and not c['elseif'] and inside(c, p)]
        if conds:
            r = chain_root(min(conds, key=lambda c: c['start'])); roots[r['start']] = r
    for r in roots.values():
        out.append((line(r['start']), 'C', r['start'], chain_end(r)))
    loops = {}
    for m in re.finditer(r'\b(continue|break|return)\b', s):
        p = m.start()
        enc = sorted([c for c in cons if c['kw'] in ('loop', 'switch') and inside(c, p)], key=lambda c: -c['start'])
        if m.group(1) == 'break': lp = enc[0] if enc and enc[0]['kw'] == 'loop' else None
        else: lp = next((c for c in enc if c['kw'] == 'loop'), None)
        if lp and has_bar(lp): loops[lp['start']] = lp
    for lp in loops.values():
        out.append((line(lp['start']), 'L', lp['start'], lp['be']))
    for c in cons:
        if c['kw'] == 'loop' and re.search(r'sgitg|tiisg|simd_gid|simd_lid|tpitg', c['head']) and has_bar(c):
            out.append((line(c['start']), 'H', c['start'], c['be']))
    res = []
    for ln, rule, a, e in out:
        text = ' '.join(s[a:e + 1].split())
        nl = s.find('\n', a)
        res.append((rel, ln, rule, s[a:nl if nl >= 0 else len(s)].strip(), hashlib.sha256(text.encode()).hexdigest()[:12]))
    return res

finds = []
for d, _, fs in os.walk(root):
    for f in fs:
        if f.endswith('.metal'):
            path = os.path.join(d, f)
            with open(path, encoding='utf-8') as fh:
                finds += scan(os.path.relpath(path, root), strip(fh.read()))
finds.sort(key=lambda x: (x[0], x[1], x[2]))
rows = []
with open(rows_path, encoding='utf-8') as fh:
    for ln in fh:
        if ln.strip() and not ln.startswith('#'):
            f = ln.rstrip('\n').split('\t')
            if len(f) != 6 or f[4] not in ('reviewed', 'open'):
                print('BAD-ROW ' + ln.rstrip('\n')); sys.exit(1)
            rows.append(f + [False])
bad = 0; nrev = nopen = 0
for rel, ln, rule, first, h in finds:
    r = next((r for r in rows if not r[6] and r[:4] == [rel, rule, first, h]), None)
    if r is None:
        print('BARRIER %s %s:%d %s %s' % (rule, rel, ln, h, first)); bad += 1; continue
    r[6] = True
    if r[4] == 'open':
        nopen += 1; print('OPEN %s:%d %s' % (rel, ln, r[5]))
    else:
        nrev += 1
for r in rows:
    if not r[6]:
        print('STALE-ROW ' + '\t'.join(r[:4])); bad += 1
print('barrier loops: %d findings, %d reviewed, %d open' % (len(finds), nrev, nopen))
sys.exit(1 if bad else 0)
EOF
