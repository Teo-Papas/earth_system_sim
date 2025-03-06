# GitHub Quick Reference Guide

Quick guide for common GitHub operations with the Earth System Simulation project.

## First Time Setup

```bash
# Clone the repository
git clone https://github.com/your-username/earth_system_sim.git
cd earth_system_sim

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run setup script
python scripts/setup_github.py --token YOUR_GITHUB_TOKEN
```

## Daily Development

### 1. Start Working

```bash
# Get latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Stage changes
git add .

# Commit (pre-commit hooks will run automatically)
git commit -m "[Component] Brief description of changes"

# Push to GitHub
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub repository
2. Click "Compare & pull request"
3. Fill in description
4. Request review

## Common Tasks

### Update Dependencies

```bash
# Update requirements
pip freeze > requirements.txt

# Update pre-commit
pre-commit autoupdate
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_integration.py

# Run with coverage
pytest --cov=src/ tests/
```

### Format Code

```bash
# Run pre-commit on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

## Troubleshooting

### Push Rejected

```bash
# Update local repository
git pull --rebase origin main

# Force push if needed (use with caution!)
git push -f origin feature/your-feature-name
```

### Pre-commit Fails

```bash
# Fix formatting issues
black .
isort .

# Retry commit
git commit -m "Your message"
```

### Merge Conflicts

```bash
# Update branch
git fetch origin
git rebase origin/main

# Resolve conflicts and continue
git add .
git rebase --continue
```

## Quick Commands Reference

### Branch Management

```bash
# List branches
git branch

# Switch branch
git checkout branch-name

# Delete local branch
git branch -d branch-name

# Delete remote branch
git push origin --delete branch-name
```

### Changes

```bash
# View status
git status

# View changes
git diff

# Undo last commit
git reset --soft HEAD^

# Discard changes
git restore file-name
```

### Remote Operations

```bash
# View remotes
git remote -v

# Add remote
git remote add origin https://github.com/username/repo.git

# Update remote URL
git remote set-url origin new-url
```

### Tags

```bash
# Create tag
git tag -a v1.0.0 -m "Version 1.0.0"

# Push tags
git push origin --tags

# Delete tag
git tag -d tag-name
```

## GitHub Actions

### View Workflows

1. Go to repository
2. Click "Actions" tab
3. Select workflow

### Re-run Failed Jobs

1. Open failed workflow
2. Click "Re-run jobs"

### Skip CI

Add `[skip ci]` to commit message:
```bash
git commit -m "[Component] Update docs [skip ci]"
```

## Best Practices

1. **Branch Names**
   - `feature/` for new features
   - `bugfix/` for bug fixes
   - `docs/` for documentation
   - `test/` for testing

2. **Commit Messages**
   - Start with component in brackets
   - Use present tense
   - Be descriptive but concise

3. **Pull Requests**
   - Reference issues
   - Include test coverage
   - Update documentation
   - Add screenshots if UI changes

4. **Code Review**
   - Respond to all comments
   - Test changes locally
   - Update PR as needed

## Resources

- [Full GitHub Setup Guide](GITHUB_SETUP.md)
- [GitHub Documentation](https://docs.github.com)
- [Pre-commit Documentation](https://pre-commit.com)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)