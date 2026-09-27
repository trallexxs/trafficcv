"""Hugging Face Space entry point: fetches the team repository and serves its website (app.py)."""
import os
import subprocess
import sys

REPO = os.environ.get("REPO_URL", "https://github.com/OWNER/REPO")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")

if not os.path.isdir(CODE):
    subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
sys.path.insert(0, CODE)
os.chdir(CODE)

import app as site  # noqa: E402  (the repository's app.py)

site.demo.queue(max_size=8).launch(css=site.CSS, theme=site.gr.themes.Soft())
