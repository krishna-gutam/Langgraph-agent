import os
import pytest
from tools.shell_exec.shell_exec import run_bash

def test_run_bash_success():
    result = run_bash.invoke({"script": "echo 'Hello from bash'", "justification": "Testing basic command execution"})
    assert "Hello from bash" in result

def test_run_bash_nonzero_exit():
    result = run_bash.invoke({"script": "exit 42", "justification": "Testing non-zero exit code handling"})
    assert "Error (exit code 42)" in result

def test_run_bash_venv_and_python():
    # Verify that python runs from the virtual environment and pip/python functions are available
    script = "python -c 'import sys; print(sys.executable)'"
    result = run_bash.invoke({"script": script, "justification": "Testing venv python execution"})
    assert ".venv" in result
