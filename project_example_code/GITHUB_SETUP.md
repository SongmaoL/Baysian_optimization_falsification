# GitHub Setup Guide

Follow these steps to create a GitHub repository and push this project.

## Step 1: Create GitHub Repository

1. Go to https://github.com and log in
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in repository details:
   - **Repository name**: `carla-falsification-framework` (or your choice)
   - **Description**: `Multi-Objective Bayesian Optimization for CARLA Falsification`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README (we already have one)
4. Click **"Create repository"**

## Step 2: Initialize Local Git Repository

Open a terminal in your project directory and run:

```bash
# Navigate to project directory
cd C:\Users\Matthew\Desktop\hw_fall_2025\project_example_code

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Multi-objective falsification framework for CARLA"
```

## Step 3: Connect to GitHub

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual values:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### If You Get Authentication Errors

GitHub requires a Personal Access Token (PAT) instead of password:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Generate and **copy the token** (you won't see it again!)
5. When pushing, use the token as your password:
   ```bash
   Username: YOUR_GITHUB_USERNAME
   Password: YOUR_PERSONAL_ACCESS_TOKEN
   ```

### Alternative: Use SSH

If you prefer SSH:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to GitHub
# Copy the public key: cat ~/.ssh/id_ed25519.pub
# Go to GitHub → Settings → SSH and GPG keys → New SSH key
# Paste the key

# Use SSH remote
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## Step 4: Verify Upload

1. Go to your GitHub repository page
2. You should see all files:
   - README.md (rendered as homepage)
   - All Python files
   - Documentation files
   - CARLA mini-project

## Step 5: Add Collaborators (Optional)

If working with a team:

1. Go to repository → Settings → Collaborators
2. Add team members by username or email

## Step 6: Create Release (Optional)

To create a version release:

1. Go to repository → Releases → Draft a new release
2. Create tag: `v1.0.0`
3. Title: `Initial Release - Multi-Objective Falsification Framework`
4. Description: Add notes about features
5. Publish release

## GitHub Repository Structure

Your repository will look like:

```
carla-falsification-framework/
├── .gitignore
├── LICENSE
├── README.md                    ← GitHub homepage
├── QUICKSTART.md
├── IMPLEMENTATION_SUMMARY.md
├── requirements.txt
├── project_proposal.md
│
├── config/                      # Search space
├── metrics/                     # Objectives
├── analysis/                    # Visualization
│
├── scenario_generator.py
├── multi_objective_bo.py
├── falsification_framework.py
│
└── csci513-miniproject1/        # CARLA project
```

## Updating the Repository

After making changes:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Common Git Commands

```bash
# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Merge branch
git merge feature-name

# Pull latest changes
git pull

# View differences
git diff
```

## Troubleshooting

### "Failed to push refs"

```bash
# Pull first, then push
git pull origin main --rebase
git push origin main
```

### "Large files detected"

GitHub has a 100MB file size limit. Check `.gitignore` excludes:
- `*.mp4` (videos)
- `*.csv` (large logs)
- CARLA wheel files (if too large)

### "Permission denied"

- Check your GitHub credentials
- Verify you have write access to the repository
- Re-generate Personal Access Token if expired

## Best Practices

1. **Commit often** with clear messages
2. **Use branches** for new features
3. **Write good commit messages**: 
   - Good: "Add weather parameter validation"
   - Bad: "fix stuff"
4. **Don't commit** large files or sensitive data
5. **Update README** when adding features

## Making the Repository Public

If you want to share with others:

1. Go to repository → Settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Make public"
4. Confirm

## Adding Badges to README

Add these badges to the top of `README.md`:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)
```

## Done! 🎉

Your project is now on GitHub and ready to share!

**Repository URL**: `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`

Share this URL with:
- Team members
- Instructors
- On your resume/portfolio
- In research papers

---

**Need help?** Check [GitHub Docs](https://docs.github.com/) or ask your team!

