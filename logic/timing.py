import time
from collections import defaultdict
from contextlib import contextmanager

# Per-process only. Pool workers get their own copy, not shared with main.
TOTALS = defaultdict(float)
COUNTS = defaultdict(int)


@contextmanager
def timed(label):
    t = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t
        TOTALS[label] += dt
        COUNTS[label] += 1


def report(reset=True):
    print("\n=== TIMING REPORT ===")
    for k, v in sorted(TOTALS.items(), key=lambda x: -x[1]):
        n = COUNTS[k]
        print(f"TIMING {k:<22} total={v:8.2f}s  calls={n:5d}  avg={v / n * 1000:8.1f}ms")
    if reset:
        TOTALS.clear()
        COUNTS.clear()
