"""Validate chlorotoxin wild-type copies against the audited frozen positions."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2 import mask_builder


ANCHOR = "CDDCCGGKGRGKCYGPQCLCR"
ASSIGNMENT_PATTERN = re.compile(
    r"\bWILD_TYPE\s*(?::\s*[^=]+)?=\s*(['\"])([A-Za-z]+)\1"
)
QUOTED_PATTERN = re.compile(r"(['\"])([A-Za-z]+)\1")


def derive_ground_truth_positions():
    position_maps = []
    for name, value in vars(mask_builder).items():
        if not isinstance(value, dict) or not value:
            continue
        if all(
            isinstance(position, int)
            and isinstance(residue, str)
            and len(residue) == 1
            for position, residue in value.items()
        ):
            position_maps.append((name, value))

    ground_truth_positions = {}
    for name, position_map in position_maps:
        for position, residue in position_map.items():
            previous = ground_truth_positions.get(position)
            if previous is not None and previous != residue:
                raise ValueError(
                    f"Conflicting mask definitions at position {position}: "
                    f"{previous!r} versus {residue!r} in {name}"
                )
            ground_truth_positions[position] = residue
    return dict(sorted(ground_truth_positions.items()))


def run_grep(pattern):
    command = ["grep", "-rn", pattern, "--include=*.py", "."]
    try:
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.strip() or "grep failed")
        return result.stdout.splitlines()
    except FileNotFoundError:
        result = subprocess.run(
            ["git", "grep", "-n", pattern, "--", "*.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.strip() or "git grep failed")
        return result.stdout.splitlines()


def parse_search_results(lines):
    copies = {}
    for line in lines:
        path, line_number, source = line.split(":", 2)
        path = path.removeprefix("./")

        assignment = ASSIGNMENT_PATTERN.search(source)
        if assignment:
            copies[(path, int(line_number))] = assignment.group(2)

        if ANCHOR in source:
            for quoted in QUOTED_PATTERN.findall(source):
                sequence = quoted[1]
                if ANCHOR in sequence:
                    copies[(path, int(line_number))] = sequence
    return copies


def discover_target_copies():
    wild_type_matches = parse_search_results(run_grep("WILD_TYPE"))
    anchor_matches = parse_search_results(run_grep(ANCHOR))
    discovered = {**wild_type_matches, **anchor_matches}
    expected_length = len(mask_builder.EXPECTED_SEQUENCE)
    return {
        location: sequence
        for location, sequence in discovered.items()
        if len(sequence) == expected_length or ANCHOR in sequence
    }


def validate(copies, ground_truth_positions):
    errors = []
    for (path, line_number), sequence in sorted(copies.items()):
        for position, expected in ground_truth_positions.items():
            actual = sequence[position] if position < len(sequence) else "<missing>"
            if actual != expected:
                errors.append(
                    f"{path}:{line_number} position {position}: "
                    f"expected {expected!r}, found {actual!r}"
                )

    locations = sorted(copies)
    if locations:
        reference_location = locations[0]
        reference_sequence = copies[reference_location]
        for location in locations[1:]:
            sequence = copies[location]
            if sequence != reference_sequence:
                first_difference = next(
                    (
                        position
                        for position, (left, right) in enumerate(
                            zip(reference_sequence, sequence)
                        )
                        if left != right
                    ),
                    min(len(reference_sequence), len(sequence)),
                )
                errors.append(
                    f"{location[0]}:{location[1]} differs from "
                    f"{reference_location[0]}:{reference_location[1]} at position "
                    f"{first_difference}: expected {reference_sequence[first_difference:first_difference + 1]!r}, "
                    f"found {sequence[first_difference:first_difference + 1]!r}"
                )
    return errors


def main():
    ground_truth_positions = derive_ground_truth_positions()
    copies = discover_target_copies()
    if not copies:
        print("ERROR: no target wild-type copies discovered")
        return 1

    errors = validate(copies, ground_truth_positions)
    if errors:
        print("Wild-type consistency check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Checked {len(copies)} wild-type copies across "
        f"{len({path for path, _ in copies})} files - all consistent with "
        "mask_builder.py and with each other."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
