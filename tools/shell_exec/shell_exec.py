import os
import shlex
import shutil
import subprocess
import sys
from langchain_core.tools import tool

TIMEOUT_SECONDS = 600


def _find_binary(binary_dir: str, names: list[str]) -> str | None:
    for name in names:
        candidate = os.path.join(binary_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


@tool
def run_bash(script: str, justification: str) -> str:
    """Executes a bash command.

    Args:
        script: The bash command or script to execute.
        justification: Explain why this command needs to be executed.
    """

    bash_exe = shutil.which("bash")
    if not bash_exe:
        return "Error: 'bash' was not found on PATH."

    current_workspace = os.getcwd()
    venv_dir = os.path.join(current_workspace, ".venv")
    venv_unix = os.path.join(venv_dir, "bin")
    venv_win = os.path.join(venv_dir, "Scripts")  # Git Bash / MSYS on Windows

    # 1. Create the .venv if it doesn't exist
    if not os.path.isdir(venv_dir):
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", ".venv"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or e.stdout or "").strip() or str(e)
            return f"Error: Failed to automatically create a '.venv'.\n{detail}"

    # 2. Locate the interpreter and pip explicitly
    if os.path.isdir(venv_unix):
        active_venv = venv_unix
    elif os.path.isdir(venv_win):
        active_venv = venv_win
    else:
        return "Error: Virtual environment created, but binary folders are missing."

    python_exe = _find_binary(active_venv, ["python", "python3", "python.exe"])
    pip_exe = _find_binary(active_venv, ["pip", "pip3", "pip.exe"])

    if not python_exe or not pip_exe:
        return (
            f"Error: '.venv' is incomplete — could not locate "
            f"{'python' if not python_exe else 'pip'} in {active_venv}."
        )

    # 3. ZERO-FALLBACK ENFORCEMENT:
    # Aliases are ignored in non-interactive bash, so shadow python/pip with
    # exported functions. `set -e` makes any failure hard-crash the script.
    q = shlex.quote
    prelude = [
        "set -euo pipefail",
        f"export VIRTUAL_ENV={q(venv_dir)}",
        f"export PATH={q(active_venv)}:$PATH",
        "unset PYTHONHOME",
        f"python() {{ command {q(python_exe)} \"$@\"; }}",
        f"python3() {{ command {q(python_exe)} \"$@\"; }}",
        f"pip() {{ command {q(pip_exe)} \"$@\"; }}",
        f"pip3() {{ command {q(pip_exe)} \"$@\"; }}",
        "export -f python python3 pip pip3",
        "hash -r",
    ]
    injected_script = "\n".join(prelude) + "\n" + script

    try:
        result = subprocess.run(
            [bash_exe, "-c", injected_script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=current_workspace,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: Command timed out after {TIMEOUT_SECONDS} seconds. "
            "It may be stuck in an infinite loop or waiting on input."
        )

    output = result.stdout.strip()
    error = result.stderr.strip()

    if result.returncode != 0:
        detail = error or output or "(no output)"
        return f"Error (exit code {result.returncode}): {detail}"

    if output:
        return output
    if error:
        # Non-fatal stderr (warnings, progress bars, etc.)
        return f"Command succeeded with stderr output:\n{error}"
    return "Command executed successfully (no output)."