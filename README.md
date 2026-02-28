# Focal Cross Entropy package

[![Publish to PyPI](https://github.com/RPetras3/focal_ce_loss_funcs_pytorch/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/RPetras3/focal_ce_loss_funcs_pytorch/actions/workflows/publish-to-pypi.yml)

This repository is an implementation of the Focal Crossentropy loss function that is meant to be used as a Pytorch compatible loss function using nn.Module

## Quick start

1. Fork this repository on GitHub and mark your fork as a template via the repo settings (once)
2. Try and/or remove the demo files from your fork: `src/lorenz.py` & `notebooks/lorenz_demo.ipynb` (once)
3. Create a pending publisher on [PyPI.org](https://pypi.org) (each new project)
4. Create a new repository for your project from your template fork on GitHub (each new project)
5. Update `pyproject.toml` to reflect your new project's metadata (each new project)
6. Commit and push your code (each code update)
7. Tag and publish a release on GitHub - this will trigger the workflow to publish the current commit to PyPI (each code update)

## Project structure

```
.
├── .devcontainer/               # Dev container config
├── .git/                        # Local Git metadata
├── .github/                     # GitHub configuration
│   └── workflows/
│       └── publish-to-pypi.yml  # Builds package and publishes to PyPI
├── .gitignore                   # Files and folders Git should ignore
├── LICENSE                      # Package license
├── pyproject.toml               # Package metadata, build system, and dependencies
├── README.md                    # Project overview and setup instructions
├── requirements.txt             # Dev/test dependencies for local usage
├── notebooks/
│   └── focal_cross_entropy_demo.ipynb        # Example notebook demonstrating the package
└── src/
	└── focal_ce_loss.py                # Example module shipped in the package
```
## Usage

### 1. Fork and clone the repo (once)

Fork the repo on GitHub and clone it to your local machine:

```bash
git clone https://github.com/<your_user_name>/focal_ce_loss_funcs_pytorch.git
```

### 2. Run the demo files to download the dataset and look at the example notebook.

Open `notebooks/focal_cross_entropy_demo.ipynb` and run the cells to see an example of how the loss function compares to other inbuilt loss functions in pytorch.

### 3. Make any changes you wish to the codebase, and commit and push those changes to GitHub.

```bash
git add .
git commit -m "Your commit message here"
git push origin main
```

### 4. (Optional) Create a pull request to the original repo if you think your changes would be useful to others.

On GitHub, navigate to your forked repo and click the "Compare & pull request" button.
- Follow the prompts to create a pull request against the original repository.
- The maintainers of the original repo will review your changes.

### 5. Update project metadata (each new project)

Update `pyproject.toml` with your project metadata:

- Change `name`, `authors`, and `description` in `[tool.poetry]`
- Update `classifiers` and `keywords` in `[tool.poetry]`
- Update runtime dependencies in `[tool.poetry.dependencies]`
- Replace or remove `[tool.poetry.urls]`

### 6. Commit and push (each code update)

Use the repo to develop your project as you normally would. Place Python modules to be published as part of your package in the `src/` directory. Commit and push your code to GitHub.

### 7. Tag and publish a release (each code update)

1. Create a GitHub release tag that matches your current `version` under `[tool.poetry]` in `pyproject.toml` (e.g., `0.1.0` or `v0.1.0` - see [here](https://semver.org/) for information about version numbering).
2. The **Publish to PyPI** workflow will build and publish automatically.

If you prefer a manual run, you can trigger the workflow from the **Actions** tab.
