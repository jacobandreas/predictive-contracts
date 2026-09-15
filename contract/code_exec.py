"""Run untrusted Python in a resource-limited subprocess and check assertions.

Adapted (and simplified) from ariahw/rl-rewardhacking `src/evaluate`.  The
program and the test assertions are executed together in one fresh Python
process with memory and CPU-time limits; the process prints a JSON summary of
how many assertions passed.
"""
import json
import os
import subprocess
import sys
import textwrap

_RUNNER = textwrap.dedent(
    """
    import json, os, resource, signal, sys
    memory_bytes = int(sys.argv[1]) * 1024 * 1024
    time_limit = int(sys.argv[2])
    for lim in (resource.RLIMIT_AS, resource.RLIMIT_RSS):
        try:
            resource.setrlimit(lim, (memory_bytes, memory_bytes))
        except (ValueError, OSError):
            pass
    class Timeout(Exception):
        pass
    def on_alarm(signum, frame):
        raise Timeout()
    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(time_limit)

    src = sys.stdin.read()
    out = {"compiled": True, "timeout": False, "error": None, "passed": 0, "evaluated": 0, "errors": []}
    # Anything the program prints is discarded (a program printing in a loop can otherwise produce
    # gigabytes before the time limit fires); only the JSON summary below goes to the real stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        code = compile(src, "<program>", "exec")
    except (SyntaxError, IndentationError) as e:
        out["compiled"] = False
        out["error"] = type(e).__name__ + ": " + str(e)
        sys.stdout = real_stdout
        print("\\n" + json.dumps(out)); sys.exit(0)
    ns = {}
    try:
        exec(code, ns)
    except Timeout:
        out["timeout"] = True
    except BaseException as e:
        out["error"] = type(e).__name__ + ": " + str(e)[:200]
    # The program may define __tests__ (a list of assertion strings) which we run one by one.
    for t in ns.get("__tests__", []):
        out["evaluated"] += 1
        try:
            exec(t, ns)
            out["passed"] += 1
        except Timeout:
            out["timeout"] = True
            break
        except BaseException as e:
            out["errors"].append(type(e).__name__ + ": " + str(e)[:200])
            break  # stop at the first failing test (as in the original repo)
    signal.alarm(0)
    sys.stdout = real_stdout
    print("\\n" + json.dumps(out))
    """
).strip()


def run_program(src, timeout=3, memory_mb=1024):
    """Execute `src` in a subprocess. Returns the JSON summary dict printed by the runner."""
    try:
        p = subprocess.run(
            [os.path.realpath(sys.executable), "-c", _RUNNER, str(memory_mb), str(timeout)],
            input=src,
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"compiled": True, "timeout": True, "error": "hard timeout", "passed": 0, "evaluated": 0, "errors": []}
    except (json.JSONDecodeError, IndexError):
        return {"compiled": False, "timeout": False, "error": "runner crashed (OOM?)", "passed": 0, "evaluated": 0, "errors": []}


def run_tests(setup_code, program, tests, timeout=3):
    """Run `program` (after `setup_code`) and then each assertion string in `tests`.

    Returns a dict with `compiled`, `passed`, `total`, `pass_rate`, `all_pass`, `errors`.
    """
    src = f"{setup_code}\n\n{program}\n\n__tests__ = {tests!r}\n"
    r = run_program(src, timeout=timeout)
    total = len(tests)
    passed = r["passed"] if r["compiled"] and r["error"] is None else 0
    return {
        "compiled": r["compiled"],
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "all_pass": total > 0 and passed == total,
        "errors": ([r["error"]] if r["error"] else []) + r["errors"] + (["Timeout"] if r["timeout"] else []),
    }
