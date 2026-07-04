"""
Span detector + collapse policy for CTA.

Finds repeated token-id spans in a single sequence (session-scoped),
then selects which repeats to collapse based on policy:
  freq(s) >= f_min  AND  len(s) >= k_min

Uses rolling-hash over fixed window sizes to find repeated substrings cheaply.
Returns a set of NON-OVERLAPPING spans to collapse (greedy: longest & most frequent first),
keeping the FIRST occurrence expanded (anchor) and collapsing later duplicates.
Actually for compute-saving we collapse ALL occurrences (each duplicate -> 1 token).
"""
from collections import defaultdict


def find_repeated_spans(token_ids, k_min=3, k_max=16, f_min=2):
    """Return dict: span_tuple -> list of start indices (occurrences), for repeats."""
    n = len(token_ids)
    candidates = {}
    for k in range(k_max, k_min - 1, -1):  # prefer longer spans
        seen = defaultdict(list)
        for i in range(n - k + 1):
            key = tuple(token_ids[i:i + k])
            seen[key].append(i)
        for key, starts in seen.items():
            if len(starts) >= f_min:
                candidates[key] = starts
    return candidates


def select_collapse_spans(token_ids, k_min=3, k_max=16, f_min=2):
    """Greedy non-overlapping selection. Returns list of (start, end_exclusive) to collapse.

    Score = length * (freq) — favors long, frequent repeats. Keeps positions disjoint.
    """
    cands = find_repeated_spans(token_ids, k_min, k_max, f_min)
    # Flatten to (score, length, start, end) occurrences
    occ = []
    for key, starts in cands.items():
        k = len(key)
        freq = len(starts)
        for s in starts:
            occ.append((k * freq, k, s, s + k))
    # Sort: highest score, then longest, then earliest
    occ.sort(key=lambda t: (-t[0], -t[1], t[2]))
    used = [False] * len(token_ids)
    chosen = []
    for score, k, s, e in occ:
        if any(used[s:e]):
            continue
        for j in range(s, e):
            used[j] = True
        chosen.append((s, e))
    chosen.sort()
    return chosen


def build_segments(token_ids, collapse_spans):
    """Return ordered list of segments covering the whole sequence.
    Each segment: (start, end_exclusive, is_collapsed).
    Non-collapsed regions are emitted as single-token segments (is_collapsed=False).
    """
    n = len(token_ids)
    collapse_set = {}
    for (s, e) in collapse_spans:
        collapse_set[s] = e
    segments = []
    i = 0
    while i < n:
        if i in collapse_set:
            e = collapse_set[i]
            segments.append((i, e, True))
            i = e
        else:
            segments.append((i, i + 1, False))
            i += 1
    return segments


if __name__ == "__main__":
    # Toy: repeated identifiers / boilerplate
    seq = [10, 20, 30, 40, 10, 20, 30, 99, 50, 10, 20, 30, 77]
    spans = select_collapse_spans(seq, k_min=3, k_max=6, f_min=2)
    print("collapse spans:", spans)
    segs = build_segments(seq, spans)
    n_collapsed = sum(1 for _,_,c in segs if c)
    print(f"segments ({len(segs)}): {segs}")
    print(f"original len={len(seq)} -> collapsed len={len(segs)}  ({n_collapsed} collapsed spans)")
