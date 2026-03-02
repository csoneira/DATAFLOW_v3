# Assets Directory Structure

This directory is organized to keep documentation figures traceable and maintainable.

## Subdirectories

- `figures/architecture/`: system and workflow diagrams used across software and operations pages.
- `figures/diagnostics/`: plots and analysis diagnostics copied from repository outputs.
- `figures/maps/`: network and station-location maps.
- `photos/collaboration/`: workshop and conference collaboration photos.
- `photos/minitrasgo/`: detector build, deployment, and maintenance photos.
- `logos/`: project and collaboration logos used by MkDocs theme and pages.
- `js/`: documentation-specific JavaScript (e.g., Mermaid init).

## Usage conventions

- Prefer embedding only representative images in pages; avoid large photo dumps.
- Keep source/provenance notes in [references/figure-gallery.md](../references/figure-gallery.md) for nontrivial figures.
- Use relative links from markdown pages (for example `../assets/figures/maps/europe_network.png`) or site-root paths (`/assets/...`) where appropriate.
