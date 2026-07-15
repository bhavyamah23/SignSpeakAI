import csv

input_file = "../dataset/gestures.csv"
output_file = "../dataset/gestures_clean.csv"

with open(input_file, "r", newline="") as infile, \
     open(output_file, "w", newline="") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    kept = 0
    removed = 0

    for row in reader:

        # Keep only correct rows
        if len(row) == len(header):
            writer.writerow(row)
            kept += 1
        else:
            removed += 1

print(f"Rows Kept    : {kept}")
print(f"Rows Removed : {removed}")

print("\nClean dataset saved as:")
print(output_file)