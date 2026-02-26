# miniTRASGO Documentation

This MkDocs documentation is part of the main `DATAFLOW_v3` repository and lives in `DOCUMENTATION/`.

## Location in this repository

```text
DATAFLOW_v3/
├── DOCUMENTATION/
│   ├── mkdocs.yml
│   ├── requirements.txt
│   └── docs/
└── .github/workflows/deploy-documentation.yml
```

## Contributing

Contributions are made through pull requests against the main repository.

1. Fork the main repository.
2. Edit Markdown pages under `DOCUMENTATION/docs/` and, when needed, update navigation in `DOCUMENTATION/mkdocs.yml`.
3. Open a pull request targeting the default branch.

You can also use GitHub's web editor for quick text fixes in `DOCUMENTATION/docs/`.

## Build and serve locally

```bash
cd DOCUMENTATION
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

To build the static site locally:

```bash
cd DOCUMENTATION
source .venv/bin/activate
mkdocs build
```

The generated site is written to `DOCUMENTATION/site/`.

## Deployment Process

The documentation site is deployed from this repository via GitHub Actions and GitHub Pages.

- Workflow: `.github/workflows/deploy-documentation.yml`
- Trigger: push to `main` and manual `workflow_dispatch`
- Build source: `DOCUMENTATION/`
- Published artifact: `DOCUMENTATION/site/`
- Hosted URL: <https://csoneira.github.io/DATAFLOW_v3/>

## Troubleshooting

- Ensure Markdown syntax is valid.
- Ensure new pages are included in `DOCUMENTATION/mkdocs.yml` navigation.
- If deployment fails, inspect the latest GitHub Actions run for the workflow above.

## Contact

For issues or suggestions, open an issue in this repository.
