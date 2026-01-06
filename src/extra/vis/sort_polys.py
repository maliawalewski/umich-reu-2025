import numpy as np

DATA_PATH   = "samples.npz"
VAR_NAMES   = ("x", "y", "z")
MAX_SAMPLES = None
MAX_TERMS   = None 
NORMALIZE_W = False 

def grevlex_key(exp_xyz):
    a, b, c = (int(t) for t in exp_xyz)
    deg = a + b + c
    return (-deg, c, b, a)


def wgrevlex_key(exp_xyz, w):
    e = np.asarray(exp_xyz, dtype=float)
    w = np.asarray(w, dtype=float)
    wdeg = float(e @ w)
    deg = int(e.sum())
    a, b, c = (int(t) for t in e)
    return (-wdeg)


def format_monom(exp_xyz, var_names=VAR_NAMES):
    a, b, c = (int(t) for t in exp_xyz)
    parts = []
    for v, p in zip(var_names, (a, b, c)):
        if p == 0:
            continue
        if p == 1:
            parts.append(f"{v}")
        else:
            parts.append(f"{v}^{p}")
    return "1" if not parts else "".join(parts)


def sort_terms(poly_terms, key_fn):
    terms = [tuple(map(int, t)) for t in np.asarray(poly_terms, dtype=int)]
    return sorted(terms, key=key_fn)


def poly_signature(terms_sorted, key_fn):
    return tuple(key_fn(t) for t in terms_sorted)


def side_by_side_lines(left_lines, right_lines, gap=6):
    w = max((len(s) for s in left_lines), default=0)
    out = []
    n = max(len(left_lines), len(right_lines))
    for i in range(n):
        L = left_lines[i] if i < len(left_lines) else ""
        R = right_lines[i] if i < len(right_lines) else ""
        out.append(L.ljust(w) + (" " * gap) + R)
    return out


def mark_term_diffs(terms_a, terms_b):
    n = max(len(terms_a), len(terms_b))
    marks = []
    for i in range(n):
        ta = terms_a[i] if i < len(terms_a) else None
        tb = terms_b[i] if i < len(terms_b) else None
        marks.append(" " if ta == tb else "*")
    return marks


def count_position_diffs(terms_a, terms_b):
    n = min(len(terms_a), len(terms_b))
    diffs = sum(1 for j in range(n) if terms_a[j] != terms_b[j])
    diffs += abs(len(terms_a) - len(terms_b))
    return diffs


def print_one_baseset(mm, w, idx):
    mm = np.asarray(mm, dtype=int)
    w = np.asarray(w, dtype=float)

    if NORMALIZE_W:
        s = float(w.sum())
        if s > 0:
            w = w / s

    grev_terms = []
    wgrev_terms = []
    for p in range(mm.shape[0]):
        gt = sort_terms(mm[p], grevlex_key)
        wt = sort_terms(mm[p], key_fn=lambda t, ww=w: wgrevlex_key(t, ww))
        if MAX_TERMS is not None:
            gt = gt[:MAX_TERMS]
            wt = wt[:MAX_TERMS]
        grev_terms.append(gt)
        wgrev_terms.append(wt)

    grev_poly_order = sorted(range(mm.shape[0]), key=lambda p: poly_signature(grev_terms[p], grevlex_key))
    wgrev_poly_order = sorted(
        range(mm.shape[0]),
        key=lambda p: poly_signature(wgrev_terms[p], lambda t, ww=w: wgrevlex_key(t, ww)),
    )

    print("=" * 110)
    print(f"BASESET idx={idx}")
    print(f"w = {np.round(w, 6)}   (sum={float(w.sum()):.6f})")
    print("")
    print("Polynomial order by leading terms:")
    print(f"  grevlex : {grev_poly_order}")
    print(f"  agent   : {wgrev_poly_order}")
    print("  -> CHANGED" if grev_poly_order != wgrev_poly_order else "  -> unchanged")
    print("")

    baseset_diff_total = 0
    baseset_positions_total = 0

    for p in range(mm.shape[0]):
        gt = grev_terms[p]
        wt = wgrev_terms[p]
        marks = mark_term_diffs(gt, wt)

        diff_positions = count_position_diffs(gt, wt)

        baseset_diff_total += diff_positions
        baseset_positions_total += len(gt)

        left = [f"poly {p}  grevlex:"] + [
            f"  {marks[j]} {format_monom(gt[j])}"
            for j in range(len(gt))
        ]
        right = [f"poly {p}  agent(order from w):"] + [
            f"  {marks[j]} {format_monom(wt[j])}   (w·e={float(np.dot(w, wt[j])):.4f})"
            for j in range(len(wt))
        ]

        for line in side_by_side_lines(left, right):
            print(line)

        print(f"  -> position diffs: {diff_positions}/{len(gt)}")
        if any(m == "*" for m in marks):
            first = next(j for j, m in enumerate(marks) if m == "*")
            print(f"  -> first difference at position {first} (0-based)")
        else:
            print("  -> identical term ordering")
        print("")

    print(f"BASESET total position diffs: {baseset_diff_total}/{baseset_positions_total} (out of 18 if 3x6)")
    print("")

    print("Leading monomials per polynomial:")
    for p in range(mm.shape[0]):
        lm_g = grev_terms[p][0]
        lm_w = wgrev_terms[p][0]
        changed = "   <-- changed" if lm_g != lm_w else ""
        print(
            f"  poly {p}: grevlex LM={format_monom(lm_g)}"
            f" | agent LM={format_monom(lm_w)} (w·e={float(np.dot(w, lm_w)):.4f}){changed}"
        )
    print("")


def main():
    data = np.load(DATA_PATH)
    X = data["X"]
    if "Y" not in data:
        raise KeyError(f"{DATA_PATH} must contain Y (custom weight vectors).")
    Y = data["Y"]

    n = len(X)
    stop = n if MAX_SAMPLES is None else min(int(MAX_SAMPLES), n)

    for i in range(stop):
        mm = X[i].reshape(3, 6, 3).astype(int)
        w = Y[i].astype(float)
        print_one_baseset(mm, w, i)


if __name__ == "__main__":
    main()
