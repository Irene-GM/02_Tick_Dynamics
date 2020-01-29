import csv


def reclass(row):
    id = row[0]
    place = row[1]
    date = row[2]
    tickcount = float(row[3])
    rest = row[4:]
    if tickcount != 0:
        tickcount = 1
    newrow = [id, place, date, tickcount] + rest
    return newrow



################
# Main Program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw.csv"
path_out = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw_binarized_ones.csv"

with open(path_in, "r", newline='') as r:
    with open(path_out, "w", newline="") as w:
        reader = csv.reader(r, delimiter=";")
        writer = csv.writer(w, delimiter=";")
        headers = next(reader)
        writer.writerow(headers)
        for row in reader:
            newrow = reclass(row)
            writer.writerow(newrow)



