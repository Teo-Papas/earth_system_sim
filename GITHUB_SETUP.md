# GitHub Repository Setup Guide

This guide explains how to set up and manage your Earth System Simulation project on GitHub.

## Initial Setup

### 1. Create a New Repository

1. Go to [GitHub](https://github.com)
2. Click "New" to create a new repository
3. Name it `earth_system_sim`
4. Make it public (for Binder/Colab support)
5. Don't initialize with README (we have our own)

### 2. Initialize Local Repository

```bash
# Navigate to project directory
cd earth_system_sim

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit"
```

### 3. Connect to GitHub

```bash
# Add remote repository
git remote add origin https://github.com/your-username/earth_system_sim.git

# Push to main branch
git push -u origin main
```

## Repository Structure

Ensure these files are included:
- `.gitignore` - Specifies which files Git should ignore
- `.github/workflows/tests.yml` - GitHub Actions configuration
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment specification

## GitHub Settings

### 1. Branch Protection

1. Go to repository Settings -> Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks to pass
   - Include administrators in restrictions

### 2. GitHub Actions

1. Enable Actions in repository Settings
2. Configure test workflow:
   - Automatic testing on push/PR
   - Code coverage reporting
   - Linting checks

### 3. GitHub Pages (Optional)

1. Enable GitHub Pages in Settings
2. Choose `main` branch `/docs` folder
3. Set up documentation publishing

## Development Workflow

### 1. Creating a New Feature

```bash
# Create new branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/your-feature-name
```

### 2. Updating Your Repository

```bash
# Get latest changes
git fetch origin

# Update main branch
git checkout main
git pull origin main

# Update feature branch
git checkout feature/your-feature-name
git rebase main
```

### 3. Creating Pull Requests

1. Go to repository on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill in description template
5. Request reviewers
6. Wait for CI checks

## Maintenance

### 1. Regular Updates

```bash
# Update dependencies
pip freeze > requirements.txt

# Update conda environment
conda env export > environment.yml
```

### 2. Clean-up

```bash
# Remove merged branches
git branch --merged main | grep -v '^[ *]*main$' | xargs git branch -d

# Prune remote branches
git remote prune origin
```

### 3. Version Tagging

```bash
# Create new version tag
git tag -a v1.0.0 -m "Version 1.0.0"

# Push tags
git push origin --tags
```

## Best Practices

### 1. Commit Messages

Use clear, descriptive commit messages:
```bash
# Format
git commit -m "[Component] Brief description

Detailed explanation of changes made,
why they were necessary, and any implications."
```

### 2. Branch Naming

Follow consistent branch naming:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions/modifications

### 3. Code Review

When reviewing pull requests:
1. Check code quality
2. Verify tests pass
3. Review documentation
4. Test functionality
5. Check performance impact

## Automation

### 1. GitHub Actions

Automated workflows include:
- Running tests
- Code coverage reporting
- Style checking
- Documentation building

### 2. Pre-commit Hooks

Setup pre-commit hooks:
```bash
# Install pre-commit
pip install pre-commit

# Install repository hooks
pre-commit install
```

## Troubleshooting

### 1. Push Rejected
```bash
# If push is rejected, pull latest changes
git pull --rebase origin main
git push origin main
```

### 2. Merge Conflicts
```bash
# Resolve conflicts
git status  # Check conflicted files
# Edit files to resolve conflicts
git add .
git rebase --continue
```

### 3. Reset Changes
```bash
# Soft reset (keep changes)
git reset --soft HEAD^

# Hard reset (discard changes)
git reset --hard HEAD^
```

## Additional Resources

- [GitHub Documentation](https://docs.github.com)
- [Git Cheat Sheet](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

## Support

For issues or questions:
1. Check existing GitHub Issues
2. Create new Issue with details
3. Contact repository maintainers
4. Consult project documentation