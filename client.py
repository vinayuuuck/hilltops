#!/usr/bin/env python3
import subprocess
import argparse
from collections import deque
import heapq, time


def _readline(pipe) -> str:
    """Read one line (stripped of CR/LF). Returns '' on EOF."""
    line = pipe.readline()
    if not line:
        return ""
    return line.rstrip("\r\n")


def _read_matrix(stdout):
    """Reads 'R C' then R lines of C ints. Returns (R, C, matrix[list[list[int]]])."""
    header = _readline(stdout)
    if not header:
        raise RuntimeError("EOF while expecting matrix header 'R C'")
    parts = header.split()
    if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
        raise RuntimeError(f"Expected 'R C', got: {header!r}")
    R, C = map(int, parts)

    M = []
    for _ in range(R):
        row_line = _readline(stdout)
        if not row_line:
            raise RuntimeError("EOF while reading matrix rows")
        row = [int(x) for x in row_line.split()]
        if len(row) != C:
            raise RuntimeError(
                f"Expected {C} columns, got {len(row)} in row: {row_line!r}"
            )
        M.append(row)
    return R, C, M


def to_tuple(mat):
    return tuple(x for row in mat for x in row)


def from_tuple(t, R, C):
    return [list(t[i * C : (i + 1) * C]) for i in range(R)]


def idx_to_rc(i, C):
    return divmod(i, C)


def rc_to_idx(r, c, C):
    return r * C + c


def analyze_state(state_tuple, R, C, N):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    mat = from_tuple(state_tuple, R, C)
    minval = min(state_tuple)
    minidx = state_tuple.index(minval)
    edges = [[] for _ in range(N)]
    for idx in range(N):
        r, c = idx_to_rc(idx, C)
        h = mat[r][c]
        for dr, dc in dirs:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < R and 0 <= c2 < C:
                if mat[r2][c2] < h:
                    edges[idx].append(rc_to_idx(r2, c2, C))
    rev = [[] for _ in range(N)]
    for u in range(N):
        for v in edges[u]:
            rev[v].append(u)
    q = deque([minidx])
    reachable = [False] * N
    reachable[minidx] = True
    while q:
        v = q.popleft()
        for u in rev[v]:
            if not reachable[u]:
                reachable[u] = True
                q.append(u)
    if not all(reachable):
        trapped = [i for i, ok in enumerate(reachable) if not ok]
        return {"perfect": False, "trapped": trapped, "minidx": minidx}
    memo = {}

    def longest_to_min(u):
        if u == minidx:
            return 0
        if u in memo:
            return memo[u]
        best = 0
        for v in edges[u]:
            cand = 1 + longest_to_min(v)
            if cand > best:
                best = cand
        memo[u] = best
        return best

    worst_by_node = [longest_to_min(i) for i in range(N)]
    max_worst = max(worst_by_node)
    return {
        "perfect": True,
        "max_worst": max_worst,
        "worst_by_node": worst_by_node,
        "minidx": minidx,
    }


def heuristic(state_tuple, R, C, N):
    info = analyze_state(state_tuple, R, C, N)
    return 0 if info["perfect"] else 1


def find_swaps(M, max_expansions=200000, prefer_trapped_only=True, time_limit=30):
    R = len(M)
    C = len(M[0])
    N = R * C

    start = to_tuple(M)
    start_info = analyze_state(start, R, C, N)
    if start_info["perfect"]:
        return []

    start_time = time.time()
    pq = []

    heapq.heappush(pq, (heuristic(start, R, C, N), 0, start, []))
    seen = {start: 0}
    expansions = 0

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    while pq:
        if time.time() - start_time > time_limit:
            break
        f, g, state, path = heapq.heappop(pq)
        if seen.get(state, 1e9) < g:
            continue
        info = analyze_state(state, R, C, N)
        if info["perfect"]:
            return [
                (
                    idx_to_rc(a, C)[0],
                    idx_to_rc(a, C)[1],
                    idx_to_rc(b, C)[0],
                    idx_to_rc(b, C)[1],
                )
                for (a, b) in path
            ]
        expansions += 1
        if expansions > max_expansions:
            break
        trapped = info.get("trapped", [])
        candidate_pairs = all_pairs
        if prefer_trapped_only and trapped:
            trapped_set = set(trapped)
            candidate_pairs = []
            for i in range(N):
                for j in range(i + 1, N):
                    if i in trapped_set or j in trapped_set:
                        candidate_pairs.append((i, j))
        for i, j in candidate_pairs:
            lst = list(state)
            lst[i], lst[j] = lst[j], lst[i]
            newt = tuple(lst)
            ng = g + 1
            if seen.get(newt, 1e9) <= ng:
                continue
            seen[newt] = ng
            h = heuristic(newt, R, C, N)
            heapq.heappush(pq, (ng + h, ng, newt, path + [(i, j)]))
    return None


def run_play(name: str, port: int):
    """
    Talk to the hilltops server via `nc`, play one game, and return:
        (R, C, matrix, verdict_line)

    `swaps` should be an iterable of (x1, y1, x2, y2).
    """
    # Start netcat
    proc = subprocess.Popen(
        ["nc", "127.0.0.1", str(port)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    try:
        # 1) Greeting
        greet = _readline(proc.stdout)  # e.g. "CONNECTED Hilltops Server"
        if not greet.startswith("CONNECTED"):
            raise RuntimeError(f"Unexpected greeting: {greet!r}")

        # 2) Send PLAY <name>
        proc.stdin.write(f"PLAY {name}\n")
        proc.stdin.flush()

        # 3) Read matrix
        R, C, M = _read_matrix(proc.stdout)

        # 4) Expect SEND_SWAPS
        token = _readline(proc.stdout)
        if token != "SEND_SWAPS":
            raise RuntimeError(f"Expected 'SEND_SWAPS', got: {token!r}")

        # 5) Send swaps
        swaps = find_swaps(M)

        proc.stdin.write(f"{len(swaps)}\n")
        for x1, y1, x2, y2 in swaps:
            proc.stdin.write(f"{x1} {y1} {x2} {y2}\n")
        proc.stdin.flush()

        # 6) Read verdict (e.g., "OK N max" or "NOT_OK")
        verdict = _readline(proc.stdout)

        # 7) Read DISCONNECTED (server then closes)
        disc = _readline(proc.stdout)
        if disc != "DISCONNECTED":
            # Server may close immediately after verdict; tolerate missing token
            pass

        return R, C, M, verdict

    finally:
        # Best-effort cleanup
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


def run_results(port: int):
    """
    Asks the hilltops server to send results and shutdown
    """
    proc = subprocess.Popen(
        ["nc", "127.0.0.1", str(port)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    proc.stdin.write("RESULTS\n")
    proc.stdin.flush()
    out, _ = proc.communicate()
    print(out)


if __name__ == "__main__":

    # Command line arguments
    # Run as python cient.py --name <name> --port <port>
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", default="Player", help="Player name to send in PLAY command"
    )
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument(
        "--results", action="store_true", help="Asks the server for results when true"
    )
    args = parser.parse_args()

    if args.results:
        run_results(port=args.port)
    else:
        R, C, M, verdict = run_play(args.name, port=args.port)

        # Do whatever you want with the matrix:
        # (here we just show it once for demo; remove if you want *no* output at all)
        print("R, C =", R, C)
        print("Matrix:")
        for row in M:
            print(" ".join(map(str, row)))
        print("Verdict:", verdict)
