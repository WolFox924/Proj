import csv
import sys
with open(sys.argv[1],'r') as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    header = next(reader)
    i = "response_time"
    time = []
    energy = []
    duration
    for row in reader:
        if i == "response_time":
            time.append(row[0])
            i = "energy_consumption"
        elif i == "energy_consumption":
            energy.append(row[5])
            i = "response_time"
