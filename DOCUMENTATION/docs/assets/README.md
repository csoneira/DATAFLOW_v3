# Assets Directory Structure

This directory is organized to keep documentation figures traceable and maintainable.

## Subdirectories

- `figures/analysis/efficiency/`: detector-efficiency and reconstruction diagnostics.
- `figures/network/`: station-location and collaboration-network maps.
- `photos/collaboration/`: workshop and conference collaboration photos.
- `photos/minitrasgo/`: detector build, deployment, and maintenance photos.
- `logos/`: project and collaboration logos used by MkDocs theme and pages.
- `js/`: documentation-specific JavaScript (e.g., Mermaid init).

## Usage conventions

- Prefer embedding only representative images in pages; avoid large photo dumps.
- Record generated-figure provenance in `plot_list.txt`; document manually maintained figure sources on the page that uses them.
- Name figures for their scientific content, not for the pipeline step that happened to produce them.
- Use lowercase kebab-case filenames.
- Use relative links from markdown pages (for example `../assets/figures/network/station-locations-europe.png`) or site-root paths (`/assets/...`) where appropriate.
