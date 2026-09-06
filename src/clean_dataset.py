import csv

try:
    from src.paths import RAW_DATASET_PATH, CLEAN_DATASET_PATH
except ImportError:
    from paths import RAW_DATASET_PATH, CLEAN_DATASET_PATH


def is_valid_row(row, header):

    # Column count must match
    if len(row) != len(header):
        return False

    # Gesture name must not be empty
    if not row[0].strip():
        return False

    # Every landmark value (everything after the Gesture column) must be a
    # valid number. This catches empty cells / corrupted rows that the old
    # column-count-only check silently let through.
    for value in row[1:]:
        try:
            float(value)
        except ValueError:
            return False

    return True


def main():

    with open(RAW_DATASET_PATH, "r", newline="") as infile, \
         open(CLEAN_DATASET_PATH, "w", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        kept = 0
        removed = 0

        for row in reader:
            if is_valid_row(row, header):
                writer.writerow(row)
                kept += 1
            else:
                removed += 1

    print(f"Rows Kept    : {kept}")
    print(f"Rows Removed : {removed}")
    print("\nClean dataset saved as:")
    print(CLEAN_DATASET_PATH)


if __name__ == "__main__":
    main()
