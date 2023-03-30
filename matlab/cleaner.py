import sys
file_name = sys.argv[1]
file_1 = file_name + "seq2.txt"
file_2 = file_name + "seqmat2.txt"
out_csv = file_name + "_out.csv"

f1 = open(file_1, "r")
f2 = open(file_2, "r")

d1 = {}
d2 = {}
for line in f1:
    tokens = line.split()
    d1[tokens[0]] = [float(tokens[1]), float(tokens[2])]
for line in f2:
    tokens = line.split()
    d2[tokens[0]] = float(tokens[1])

ff = open(out_csv, "w")
def to_string(vec):
    return ",".join(vec)
ff.write(to_string(["String", "Local", "Quantum", "Local + 1", "Quantum - (Local + 1)"]))
ff.write("\n")
for (ss, vec) in d1.items():
    local = vec[1]
    local_1 = vec[0]
    quantum = d2[ss]
    diff = quantum - local_1
    ff.write(to_string(["\'"+ss + "\'", str(local), str(quantum), str(local_1), str(diff)]))
    ff.write("\n")

