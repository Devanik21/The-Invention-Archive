import os
import subprocess
import sys
from datetime import datetime

def run_command(command, cwd=None):
    """Executes a shell command and checks for failure."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing command:\n{result.stderr}")
        sys.exit(result.returncode)
    print(result.stdout)

def main():
    print("=== Theoretical Research Publisher: Autonomous Engine ===")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Run numerical verification
    print("\n--- Stage 1: Numerical Verification ---")
    verify_script = os.path.join(base_dir, "verify_model.py")
    if os.path.exists(verify_script):
        run_command(["python3", "verify_model.py"], cwd=base_dir)
    else:
        print(f"Error: {verify_script} not found.")
        sys.exit(1)

    # 2. Compile LaTeX
    print("\n--- Stage 2: LaTeX Compilation ---")
    tex_file = "paper.tex"
    if os.path.exists(os.path.join(base_dir, tex_file)):
        # Run pdflatex and bibtex
        run_command(["pdflatex", tex_file], cwd=base_dir)
        run_command(["bibtex", "paper"], cwd=base_dir)
        run_command(["pdflatex", tex_file], cwd=base_dir)
        run_command(["pdflatex", tex_file], cwd=base_dir)
    else:
        print(f"Error: {tex_file} not found.")
        sys.exit(1)

    # 3. Patch Parent Readme (if applicable, in a real scenario this would modify Theoretical Research/Readme.md)
    print("\n--- Stage 3: Patching Parent Index ---")
    parent_readme = os.path.join(base_dir, "..", "Readme.md")
    if os.path.exists(parent_readme):
        with open(parent_readme, "a") as f:
            f.write(f"\n- **Theoretical-research-publisher**: Deployed on {datetime.now().strftime('%Y-%m-%d')}")
        print("Updated parent Readme.md")

    # 4. (Optional) Semantic Git Tagging
    # Note: We skip actual git tagging here to prevent sandbox disruption,
    # but print the command that would be run.
    print("\n--- Stage 4: Semantic Release Tagging ---")
    tag_name = f"v{datetime.now().strftime('%Y.%m.%d')}"
    print(f"[Simulated] git tag -a {tag_name} -m \"Release {tag_name}\"")
    print(f"[Simulated] git push origin {tag_name}")

    print("\n=== Deployment Successful ===")

if __name__ == "__main__":
    main()
