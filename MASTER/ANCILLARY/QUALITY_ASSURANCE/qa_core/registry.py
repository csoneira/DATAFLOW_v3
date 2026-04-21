"""Metadata family registry and discovery helpers for QUALITY_ASSURANCE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .task_setup import normalize_station_name

DEFAULT_TASK_IDS = (1, 2, 3, 4, 5)
_METADATA_FILENAME_RE = re.compile(r"task_(\d+)_metadata_(.+)\.csv$")


@dataclass(frozen=True)
class MetadataFamilySpec:
    """Declarative description of a Stage 1 metadata family."""

    name: str
    metadata_suffix: str
    category: str
    default_step: str | None = None
    supported_tasks: tuple[int, ...] = DEFAULT_TASK_IDS
    description: str = ""
    aliases: tuple[str, ...] = ()

    def metadata_csv_filename(self, task_id: int) -> str:
        if task_id not in self.supported_tasks:
            raise ValueError(
                f"Metadata family '{self.name}' does not support TASK_{task_id}. "
                f"Supported tasks: {self.supported_tasks}"
            )
        return f"task_{task_id}_metadata_{self.metadata_suffix}.csv"

    def metadata_relative_path(self, station: str | int, task_id: int) -> Path:
        station_name = normalize_station_name(station)
        return (
            Path("STATIONS")
            / station_name
            / "STAGE_1"
            / "EVENT_DATA"
            / "STEP_1"
            / f"TASK_{task_id}"
            / "METADATA"
            / self.metadata_csv_filename(task_id)
        )


@dataclass(frozen=True)
class DiscoveredMetadataFile:
    """Filesystem discovery record for a metadata CSV."""

    station_name: str
    task_id: int
    family_name: str
    metadata_suffix: str
    category: str | None
    path: Path
    registered: bool


METADATA_FAMILIES: tuple[MetadataFamilySpec, ...] = (
    MetadataFamilySpec(
        name="calibration",
        metadata_suffix="calibration",
        category="analysis",
        default_step="STEP_1_CALIBRATIONS",
        supported_tasks=(2,),
        description="Per-strip calibration metadata used to validate detector health.",
    ),
    MetadataFamilySpec(
        name="filtering",
        metadata_suffix="filter",
        category="analysis",
        default_step="STEP_2_FILTERING",
        description="Filter/cut fractions and purity metrics after cleaning/calibration.",
        aliases=("filter",),
    ),
    MetadataFamilySpec(
        name="trigger_types",
        metadata_suffix="trigger_type",
        category="analysis",
        default_step="STEP_6_TRIGGER_TYPES",
        description="Trigger-type rates, counts, and migration metrics.",
        aliases=("trigger_type",),
    ),
    MetadataFamilySpec(
        name="specific",
        metadata_suffix="specific",
        category="analysis",
        default_step="STEP_4_SPECIFIC",
        description="Task-specific science metadata and fit summaries.",
    ),
    MetadataFamilySpec(
        name="efficiency",
        metadata_suffix="efficiency",
        category="derived",
        description="Efficiency metadata generated downstream of the core tasks.",
    ),
    MetadataFamilySpec(
        name="naive_efficiency",
        metadata_suffix="naive_efficiency",
        category="derived",
        description="Naive efficiency estimates.",
    ),
    MetadataFamilySpec(
        name="robust_efficiency",
        metadata_suffix="robust_efficiency",
        category="derived",
        default_step="STEP_5_ROBUST_EFFICIENCY",
        description="Robust efficiency estimates.",
    ),
    MetadataFamilySpec(
        name="rate_histogram",
        metadata_suffix="rate_histogram",
        category="derived",
        default_step="STEP_4_RATE_HISTOGRAM",
        description="Rate histogram summaries.",
    ),
    MetadataFamilySpec(
        name="execution",
        metadata_suffix="execution",
        category="operational",
        description="Execution metadata for QA runs and Stage 1 processing.",
    ),
    MetadataFamilySpec(
        name="profiling",
        metadata_suffix="profiling",
        category="operational",
        description="Profiling/timing metadata.",
    ),
    MetadataFamilySpec(
        name="status",
        metadata_suffix="status",
        category="operational",
        description="Run/task status summaries.",
    ),
    MetadataFamilySpec(
        name="deep_filter",
        metadata_suffix="deep_fiter",
        category="derived",
        default_step="STEP_3_DEEP_FILTER",
        description="Deep filtering metadata. The on-disk suffix keeps the existing spelling.",
        aliases=("deep_fiter",),
    ),
)

_FAMILY_BY_TOKEN: dict[str, MetadataFamilySpec] = {}
for _spec in METADATA_FAMILIES:
    for _token in (_spec.name, _spec.metadata_suffix, *_spec.aliases):
        _FAMILY_BY_TOKEN[_token] = _spec


def metadata_family_names(*, category: str | None = None) -> list[str]:
    """Return registered family names, optionally filtered by category."""
    names = [spec.name for spec in METADATA_FAMILIES if category is None or spec.category == category]
    return sorted(names)


def get_metadata_family(name_or_suffix: str) -> MetadataFamilySpec:
    """Resolve a metadata family by canonical name, suffix, or alias."""
    token = str(name_or_suffix).strip()
    if not token:
        raise ValueError("Metadata family token cannot be empty.")
    spec = _FAMILY_BY_TOKEN.get(token)
    if spec is None:
        raise KeyError(f"Unknown metadata family '{name_or_suffix}'.")
    return spec


def parse_metadata_filename(filename: str | Path) -> tuple[int, str]:
    """Parse a Stage 1 metadata filename into task id and metadata suffix."""
    name = Path(filename).name
    match = _METADATA_FILENAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Not a recognized metadata filename: {name}")
    return int(match.group(1)), match.group(2)


def metadata_csv_filename(task_id: int, family: str | MetadataFamilySpec) -> str:
    """Build the metadata CSV filename for a task and family."""
    spec = family if isinstance(family, MetadataFamilySpec) else get_metadata_family(family)
    return spec.metadata_csv_filename(task_id)


def discover_station_metadata(
    repo_root: Path,
    station: str | int,
    *,
    task_ids: Iterable[int] | None = None,
    include_unregistered: bool = True,
) -> list[DiscoveredMetadataFile]:
    """Discover metadata CSV files present for a station."""
    station_name = normalize_station_name(station)
    task_id_filter = set(task_ids) if task_ids is not None else None
    station_root = repo_root / "STATIONS" / station_name / "STAGE_1" / "EVENT_DATA" / "STEP_1"
    if not station_root.exists():
        return []

    discovered: list[DiscoveredMetadataFile] = []
    for path in sorted(station_root.glob("TASK_*/METADATA/task_*_metadata_*.csv")):
        try:
            task_id, suffix = parse_metadata_filename(path.name)
        except ValueError:
            continue
        if task_id_filter is not None and task_id not in task_id_filter:
            continue

        spec = _FAMILY_BY_TOKEN.get(suffix)
        if spec is None:
            if not include_unregistered:
                continue
            discovered.append(
                DiscoveredMetadataFile(
                    station_name=station_name,
                    task_id=task_id,
                    family_name=suffix,
                    metadata_suffix=suffix,
                    category=None,
                    path=path,
                    registered=False,
                )
            )
            continue

        discovered.append(
            DiscoveredMetadataFile(
                station_name=station_name,
                task_id=task_id,
                family_name=spec.name,
                metadata_suffix=spec.metadata_suffix,
                category=spec.category,
                path=path,
                registered=True,
            )
        )

    return discovered
