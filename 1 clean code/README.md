# Automate formatting and linting in VS Code

## Formatting: Black

Black is an opinionated formatter for python. I want an automated formatting whenever I save a file. How to set up: 

1. pip install black
2. VS code user settings
   1. "python.formatting.provider": "black"
   2. "editor.formatOnSave": true
   3. "editor.formatOnSaveMode": "file"
3. Reload vs code and enjoy auto-formatted python files

## Linting: Flake8 

Flake8 is ready-to-use linter for python which is perfect to not worry too much about configurations. How to set up:

1. pip install flake8
2. install vs code extension: cornflakes
3. vs code user settings
   1. "cornflakes.linter.executablePath": "C:/Users/philipp.schmalen/python/Scripts/flake8.exe"

## Test it

Reload window in vs code. Open auto_format_on_save.py and save. Flake8 will squiggle lines where it spots some issues. 

# Git branching strategies

## Github flow

How to decide on your git workflow? When to branch and how to merge? There is no common solution for this as it depends on your use case. However, Github suggests a simple approach called *github flow*: https://guides.github.com/introduction/flow/

### TL;DR

1. anything in the main branch is always deployable
2. branch when developing and use descriptive names, like `refactor-authentication` or `user-content-cache-key`
4. commit early and often with clear commit messages to understand what has been developed
5. Pull requests to review, question and comment code
6. deploy from the development branch for testing

## Branching model using feature, release and hotfix branches

Here is a great article about a common branching strategy: https://nvie.com/posts/a-successful-git-branching-model/

## Merge conflicts

Git does not know which changes to push when the same lines within a file were edited. How to solve: https://docs.github.com/en/github/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts




