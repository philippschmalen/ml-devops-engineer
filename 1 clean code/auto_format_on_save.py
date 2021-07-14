"""
Play with autoformatting on save
Ensure to pip install black within your environment
"""

# test linting with an unnecessary import
# it should complain and suggest a solution
import sys


thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964,
    "okay": "This is getting way too long",
}


def hello():
    pass
