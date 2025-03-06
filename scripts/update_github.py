"""
Script to update GitHub repository with latest changes.
"""

import subprocess
import os
import sys
from pathlib import Path
import argparse

def run_command(command: str) -> str:
    """Run a shell command and return output."""
    try:
        # For Git commands with quotes, use shell=True
        if 'git commit' in command:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                shell=True
            )
        else:
            result = subprocess.run(
                command.split(),
                check=True,
                capture_output=True,
                text=True
            )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {command}: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def check_git_config():
    """Check if git user and email are configured."""
    try:
        name = run_command("git config --global user.name")
        email = run_command("git config --global user.email")
        
        if not name or not email:
            print("Error: Git user configuration missing!")
            print("Please run:")
            print('git config --global user.name "Your Name"')
            print('git config --global user.email "your.email@example.com"')
            sys.exit(1)
            
    except subprocess.CalledProcessError:
        print("Error: Git configuration check failed!")
        sys.exit(1)

def check_git_installation():
    """Check if git is installed and accessible."""
    try:
        version = run_command("git --version")
        print(f"Git version: {version}")
    except FileNotFoundError:
        print("Error: Git is not installed or not in PATH!")
        print("\nTo fix this:")
        print("1. Install Git from https://git-scm.com/download/win")
        print("2. During installation, choose 'Add to PATH'")
        print("3. Restart VS Code")
        print("\nOr if Git is installed, add it to PATH manually:")
        print("1. Search for 'Environment Variables' in Windows")
        print("2. Edit PATH variable")
        print("3. Add Git installation path (usually C:\\Program Files\\Git\\cmd)")
        sys.exit(1)

def update_repository():
    """Update GitHub repository with latest changes."""
    # Add git installation check at the start
    check_git_installation()
    check_git_config()

    # Get project root directory
    project_dir = Path(__file__).parent.parent

    # Change to project directory
    os.chdir(project_dir)

    print("\nUpdating GitHub repository...")
    print("==============================")

    # Check if we're in a git repository
    if not (project_dir / ".git").exists():
        print("Error: Not a git repository!")
        print("Please run setup_github.py first")
        sys.exit(1)

    # Get current branch
    current_branch = run_command("git branch --show-current")
    print(f"\nCurrent branch: {current_branch}")

    # Fetch latest changes
    print("\nFetching latest changes...")
    run_command("git fetch origin")

    # Check for unpushed changes
    unpushed = run_command("git log origin/main..HEAD --oneline")
    if unpushed:
        print("\nUnpushed commits:")
        print(unpushed)

    # Check for local changes
    status = run_command("git status --porcelain")
    if status:
        print("\nUncommitted changes:")
        print(status)

        # Stage all changes
        print("\nStaging changes...")
        run_command("git add .")

        # Commit changes - Fix the commit command
        print("\nCommitting changes...")
        try:
            commit_message = "Update: Package structure and import fixes"
            run_command(f"git commit -m '{commit_message}'")
        except subprocess.CalledProcessError:
            print("No changes to commit")

    # Pull latest changes with rebase
    print("\nPulling latest changes...")
    try:
        run_command("git pull --rebase origin main")
    except subprocess.CalledProcessError:
        print("Error: Conflicts detected!")
        print("Please resolve conflicts manually and then run:")
        print("git rebase --continue")
        sys.exit(1)

    # Push changes
    print("\nPushing changes to GitHub...")
    run_command("git push origin main")

    print("\nRepository updated successfully!")
    print("\nNext steps:")
    print("1. Check GitHub Actions for CI/CD status")
    print("2. Review changes on GitHub")
    print("3. Update documentation if needed")

def main():
    parser = argparse.ArgumentParser(
        description="Update GitHub repository with latest changes"
    )
    args = parser.parse_args()

    update_repository()

if __name__ == "__main__":
    main()