# Automated pep style and linting

## Black

Black is an opinionated formatter for python. I want an automated formatting whenever I save a file. How to set up: 

1. pip install black
2. VS code user settings
   1. "python.formatting.provider": "black"
   2. "editor.formatOnSave": true
   3. "editor.formatOnSaveMode": "file"
3. Reload vs code and enjoy auto-formatted python files

## Flake8 

Flake8 is ready-to-use linter for python which is perfect to not worry too much about configurations. How to set up:

1. pip install flake8
2. install vs code extension: cornflakes
3. vs code user settings
   1. "cornflakes.linter.executablePath": "C:/Users/philipp.schmalen/python/Scripts/flake8.exe"


