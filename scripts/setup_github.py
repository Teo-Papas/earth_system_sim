#!/usr/bin/env python3
"""
Script to initialize and set up GitHub repository for Earth System Simulation.
"""

import subprocess
import os
import sys
from pathlib import Path
import argparse
import json
from typing import List, Dict

def run_command(command: List[str], cwd: str = None) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def check_git_installed() -> bool:
    """Check if git is installed."""
    try:
        run_command(["git", "--version"])
        return True
    except FileNotFoundError:
        return False

def init_git_repo(project_dir: str) -> None:
    """Initialize git repository."""
    run_command(["git", "init"], cwd=project_dir)
    print("Initialized git repository")

def create_github_repo(repo_name: str, token: str) -> Dict:
    """Create GitHub repository using API."""
    import requests
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "name": repo_name,
        "description": "Earth System Simulation using hybrid AI approaches",
        "private": False,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True
    }
    
    response = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json=data
    )
    
    if response.status_code != 201:
        print(f"Error creating repository: {response.json()}")
        sys.exit(1)
        
    return response.json()

def setup_git_remotes(project_dir: str, repo_url: str) -> None:
    """Set up git remotes."""
    run_command(["git", "remote", "add", "origin", repo_url], cwd=project_dir)
    print(f"Added remote: {repo_url}")

def create_initial_commit(project_dir: str) -> None:
    """Create initial commit."""
    run_command(["git", "add", "."], cwd=project_dir)
    run_command(
        ["git", "commit", "-m", "Initial commit: Earth System Simulation"],
        cwd=project_dir
    )
    print("Created initial commit")

def push_to_github(project_dir: str) -> None:
    """Push to GitHub."""
    run_command(["git", "push", "-u", "origin", "main"], cwd=project_dir)
    print("Pushed to GitHub")

def setup_github_actions(project_dir: str) -> None:
    """Ensure GitHub Actions workflow files are in place."""
    workflows_dir = Path(project_dir) / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify workflow files exist
    if not (workflows_dir / "tests.yml").exists():
        print("Warning: GitHub Actions workflow file missing!")
        print("Please ensure .github/workflows/tests.yml is present")

def main():
    parser = argparse.ArgumentParser(
        description="Set up GitHub repository for Earth System Simulation"
    )
    parser.add_argument(
        "--token",
        help="GitHub personal access token"
    )
    parser.add_argument(
        "--repo-name",
        default="earth_system_sim",
        help="Repository name"
    )
    
    args = parser.parse_args()
    
    # Get project directory
    project_dir = str(Path(__file__).parent.parent)
    
    # Check git installation
    if not check_git_installed():
        print("Error: git is not installed!")
        print("Please install git first: https://git-scm.com/downloads")
        sys.exit(1)
    
    print("\nSetting up GitHub repository...")
    print("===============================")
    
    # Initialize repository
    if not (Path(project_dir) / ".git").exists():
        init_git_repo(project_dir)
    
    # Create GitHub repository if token provided
    if args.token:
        print("\nCreating GitHub repository...")
        repo_info = create_github_repo(args.repo_name, args.token)
        repo_url = repo_info["clone_url"]
        setup_git_remotes(project_dir, repo_url)
    else:
        print("\nNo GitHub token provided.")
        print("Please create repository manually and run:")
        print(f"git remote add origin <repository_url>")
        sys.exit(0)
    
    # Setup repository
    print("\nSetting up repository...")
    create_initial_commit(project_dir)
    setup_github_actions(project_dir)
    
    # Push to GitHub
    print("\nPushing to GitHub...")
    push_to_github(project_dir)
    
    print("\nSetup complete!")
    print(f"Repository URL: {repo_url}")
    print("\nNext steps:")
    print("1. Check repository settings at:")
    print(f"   https://github.com/{repo_info['full_name']}/settings")
    print("2. Enable GitHub Actions")
    print("3. Configure branch protection rules")
    print("4. Add collaborators if needed")

if __name__ == "__main__":
    main()