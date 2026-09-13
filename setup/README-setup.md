# Files that had to land here instead

Three files could not be written directly into their real folders, because
`.github/` and `.vscode/` are protected from remote writes on your machine.
Move them yourself, in VS Code or Explorer — one drag each:

| This file | Move it to |
|---|---|
| `github-workflow-weekly.yml` | `.github/workflows/weekly.yml` |
| `vscode-launch.json`         | `.vscode/launch.json` |
| `vscode-extensions.json`     | `.vscode/extensions.json` |

Create the `.github\workflows` and `.vscode` folders first — Windows Explorer
refuses to create a folder whose name starts with a dot, so make them from the
VS Code sidebar (New Folder), or from a terminal:

    mkdir .github\workflows
    mkdir .vscode

Then delete this `setup/` folder.

Only `weekly.yml` actually matters — it is the GitHub Actions schedule that keeps
the dashboard fresh without your PC being on. The two VS Code files are just
comfort: a Run/Debug entry that sets `PYTHONPATH` for you, and an extension
recommendation.
