import os
import subprocess
import sys
from langchain_core.tools import tool

@tool
def run_bash(script: str, justification: str) -> str:
    """Executes a Bash command using Git Bash.

    Args:
        script: The Bash command or script to execute.
        justification: Explain why this command needs to be executed.
    """
    bash_path = r"C:\Program Files\Git\bin\bash.exe"
    
    if not os.path.exists(bash_path):
        return f"Error: Git Bash not found at {bash_path}"

    venv_dir = os.path.join(os.getcwd(), ".venv")
    venv_scripts = os.path.join(venv_dir, "Scripts")
    
    # Create .venv if it doesn't exist
    if not os.path.exists(venv_dir):
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    
    # Prepare environment with .venv/Scripts in PATH
    env = os.environ.copy()
    if os.path.exists(venv_scripts):
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")

    try:
        result = subprocess.run(
            [bash_path, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if output:
            return output[:100000]
        if error:
            return f"Error: {error[:5000]}"
        return "Command executed successfully (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() or e.stdout.strip()
        return f"Error: {error_msg}"
