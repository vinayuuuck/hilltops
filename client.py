#!/usr/bin/env python3
import subprocess
import argparse


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


def your_algorithm(M):
    """
    This function gets a matrix, and returns swaps
    You only need to edit this
    """
    return [[0, 0, 2, 2], [1, 0, 1, 2]]


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
        swaps = your_algorithm(M)

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
