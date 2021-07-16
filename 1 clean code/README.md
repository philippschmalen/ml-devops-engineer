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

## Logging

Here is a common setup for logging. 

```python
import logging

logging.basicConfig(
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%d-%m %H:%M',
    filename='log/run.log',
    level=logging.INFO
)
```

It logs in `log/run.log` when running `logging_exercise.log`:

      16-07 13:09 root         ERROR    Input does not match expected type.
      16-07 13:09 root         INFO     SUCCESS: sum calculated


## Error handling

According to Google's styleguide: 

> Exceptions are allowed but must be used carefully. 

Exceptions should be raised explicitly with `raise`. It validates function input, enforces correct usage and indicates programming errors. Use `assert` to check for internal correctness. 

### Example

```python
def connect_to_next_port(self, minimum: int) -> int:
   """Connects to the next available port.

   Args:
      minimum: A port value greater or equal to 1024.

   Returns:
      The new minimum port.

   Raises:
      ConnectionError: If no available port is found.
   """
   if minimum < 1024:
   # Note that this raising of ValueError is not mentioned in the doc
   # string's "Raises:" section because it is not appropriate to
   # guarantee this specific behavioral reaction to API misuse.
   raise ValueError(f'Min. port must be at least 1024, not {minimum}.')
   port = self._find_next_open_port(minimum)
   if not port:
   raise ConnectionError(
         f'Could not connect to service on port {minimum} or higher.')
   assert port >= minimum, (
      f'Unexpected port {port} when minimum was {minimum}.')
   return port
```



# Resources

* [Github flow](https://guides.github.com/introduction/flow/)
* [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s3.10.2-error-messages)
* [Clean Code concepts adapted for machine learning and data science](https://github.com/davified/clean-code-ml) by David Tan