"""Validate a Pi raw archive and generate laptop review files without inference."""
import argparse
import json

from dtr.raw_capture import process_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        summary = process_capture(args.input, args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Raw capture processing failed: {error}\n')
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
