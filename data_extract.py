import numpy as np

def read_data_file(file_name):
    with open(file_name, "r") as f:
        raw_data = f.readlines()

    data = [ [float(x) for x in line.strip().split(',')] for line in raw_data ]

    return np.array(data)