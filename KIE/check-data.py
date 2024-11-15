import json
from os import scandir
from string import ascii_uppercase, digits, punctuation
import pandas as pd

valid_chars = ascii_uppercase + digits + punctuation + " \t\n"

csv_files = sorted(
    (f for f in scandir(r"sroie-2019\raw\box") if f.name.endswith("csv")), key=lambda f: f.name
)

for f in csv_files:
    with open(f, "r", encoding="utf-8") as fo:
        line_list = []
        for line_no, line in enumerate(fo, start=1):
            entries = line.split(",", maxsplit=8)
            for c in entries[-1]:
                if not c in valid_chars:
                    print(f"Invalid char {repr(c)} in {f.name} on line {line_no}")
            line_list.append(entries)
        df = pd.DataFrame(line_list)
        df.to_csv('sroie-2019/raw/box updated/' + f.name, index=False, header=None)

