"""
Helper script to generate links for running the project on various cloud platforms.
Supports Binder, Google Colab, and Kaggle.
"""

import argparse
import urllib.parse
from pathlib import Path
import json

def generate_binder_link(
    github_user: str,
    github_repo: str,
    branch: str = "main",
    filepath: str = None
) -> str:
    """
    Generate a Binder link for the repository.
    
    Args:
        github_user: GitHub username
        github_repo: Repository name
        branch: Branch name
        filepath: Optional path to notebook
        
    Returns:
        Binder URL
    """
    base_url = "https://mybinder.org/v2/gh"
    repo_url = f"{github_user}/{github_repo}/{branch}"
    
    if filepath:
        filepath = urllib.parse.quote(filepath)
        return f"{base_url}/{repo_url}?filepath={filepath}"
    return f"{base_url}/{repo_url}"

def generate_colab_link(
    github_user: str,
    github_repo: str,
    filepath: str,
    branch: str = "main"
) -> str:
    """
    Generate a Google Colab link for a notebook.
    
    Args:
        github_user: GitHub username
        github_repo: Repository name
        filepath: Path to notebook
        branch: Branch name
        
    Returns:
        Colab URL
    """
    base_url = "https://colab.research.google.com/github"
    return f"{base_url}/{github_user}/{github_repo}/blob/{branch}/{filepath}"

def create_kaggle_metadata(
    github_user: str,
    github_repo: str,
    kernel_title: str = "Earth System Simulation"
) -> dict:
    """
    Create Kaggle kernel metadata.
    
    Args:
        github_user: GitHub username
        github_repo: Repository name
        kernel_title: Title for the Kaggle kernel
        
    Returns:
        Dictionary of metadata
    """
    return {
        "id": f"{github_user}/{kernel_title.lower().replace(' ', '-')}",
        "title": kernel_title,
        "code_file": "examples/basic_simulation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            f"https://github.com/{github_user}/{github_repo}"
        ],
        "competition_sources": [],
        "kernel_sources": []
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate cloud platform links for the project"
    )
    parser.add_argument(
        "--github-user",
        required=True,
        help="GitHub username"
    )
    parser.add_argument(
        "--github-repo",
        default="earth_system_sim",
        help="Repository name"
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch name"
    )
    
    args = parser.parse_args()
    
    # Generate Binder links
    print("\nBinder Links:")
    print("-------------")
    
    # Basic notebook
    basic_notebook = "examples/basic_simulation.ipynb"
    binder_basic = generate_binder_link(
        args.github_user,
        args.github_repo,
        args.branch,
        basic_notebook
    )
    print(f"Basic Simulation: {binder_basic}")
    
    # Colab notebook
    colab_notebook = "examples/colab_simulation.ipynb"
    binder_colab = generate_binder_link(
        args.github_user,
        args.github_repo,
        args.branch,
        colab_notebook
    )
    print(f"Colab Version: {binder_colab}")
    
    # Generate Colab links
    print("\nGoogle Colab Links:")
    print("------------------")
    
    colab_basic = generate_colab_link(
        args.github_user,
        args.github_repo,
        basic_notebook,
        args.branch
    )
    print(f"Basic Simulation: {colab_basic}")
    
    colab_optimized = generate_colab_link(
        args.github_user,
        args.github_repo,
        colab_notebook,
        args.branch
    )
    print(f"Optimized Version: {colab_optimized}")
    
    # Generate Kaggle metadata
    print("\nKaggle Setup:")
    print("-------------")
    
    kaggle_meta = create_kaggle_metadata(
        args.github_user,
        args.github_repo
    )
    
    # Save Kaggle metadata
    kaggle_file = Path("kernel-metadata.json")
    with open(kaggle_file, "w") as f:
        json.dump(kaggle_meta, f, indent=2)
    
    print(f"Kaggle metadata saved to: {kaggle_file}")
    print("\nTo use on Kaggle:")
    print("1. Fork the repository")
    print("2. Add as a dataset on Kaggle")
    print("3. Create a new notebook using this dataset")
    
    # Create README badges
    print("\nREADME Badges:")
    print("--------------")
    
    binder_badge = f"[![Binder](https://mybinder.org/badge_logo.svg)]({binder_basic})"
    colab_badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_basic})"
    
    print("Add these badges to your README.md:")
    print(f"\n{binder_badge}")
    print(f"{colab_badge}")
    
    # Save badges to file
    badges_file = Path("cloud_badges.txt")
    with open(badges_file, "w") as f:
        f.write(f"Binder Badge:\n{binder_badge}\n\n")
        f.write(f"Colab Badge:\n{colab_badge}\n")
    
    print(f"\nBadges saved to: {badges_file}")

if __name__ == "__main__":
    main()