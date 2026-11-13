"""Attach an OSF download to ignored local paths without overwriting files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = {'.csv', '.tsv', '.xlsx', '.xls', '.rds', '.rdata', '.npz', '.npy',
                 '.parquet', '.arrow', '.feather', '.qsf', '.json', '.zip'}
SUBTREES = [
    'analysis/h4/data', 'analysis/ratings/primary/data',
    'analysis/ratings/primary/results', 'analysis/ratings/primary/R',
    'analysis/ratings/sensitivity/data', 'analysis/ratings/supplementary/results',
    'translation/materials', 'figures/embeddings',
]


def destinations(source):
    for relative in SUBTREES:
        origin = source / relative
        if origin.is_dir():
            for path in origin.rglob('*'):
                if path.is_file() and path.suffix.lower() in DATA_SUFFIXES:
                    yield path, ROOT / path.relative_to(source)
    for relative in ['analysis/scale_names.csv', 'translation/model_assignments.csv',
                     'figures/h1_ratings.csv', 'figures/h3_ratings.csv']:
        if (source / relative).is_file():
            yield source / relative, ROOT / relative
    # Compatibility with the existing coauthor export layout.
    for old, new in [('Without_Exclusion_Criteria_3', 'primary'),
                     ('With_Exclusion_Criteria_3', 'sensitivity')]:
        for original, target in [('data', 'data'), ('Analysis/results', 'results')]:
            origin = source / old / original
            if origin.is_dir():
                for path in origin.rglob('*'):
                    if path.is_file() and path.suffix.lower() in DATA_SUFFIXES:
                        yield path, ROOT / 'analysis/ratings' / new / target / path.relative_to(origin)
        if new == 'primary':
            for path in (source / old / 'Analysis').glob('*.csv'):
                yield path, ROOT / 'analysis/ratings/primary/R' / path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        parser.error('Source must be an extracted local download directory.')
    pairs = dict((dest, origin.resolve()) for origin, dest in destinations(source))
    if not pairs:
        parser.error('No matching inputs found; see docs/reproduce.md for the expected local layout.')
    # Validate all destinations before creating any links.
    for dest, origin in pairs.items():
        if dest.exists() or dest.is_symlink():
            if dest.resolve() != origin:
                parser.error(f'Refusing to overwrite {dest.relative_to(ROOT)}')
    if not args.dry_run:
        for dest, origin in pairs.items():
            if not dest.is_symlink() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(origin)
    print(f'{"Would attach" if args.dry_run else "Attached"} {len(pairs)} local input files; excluded from Git.')


if __name__ == '__main__':
    main()
