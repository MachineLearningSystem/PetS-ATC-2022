import os
import subprocess
import re
import sys

def get_output(command):
    output = subprocess.getoutput(command)
    lines = output.split("\n")
    result = []
    pat = r'\d+\.\d+|\d+'
    for line in [lines[6], lines[9], lines[13]]:
        phases = line.split(",")
        for phase in phases:
            digit = re.findall(pat, phase)
            result.append(float(digit[0]))
    return result

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python " + sys.argv[0] +
              " plug_example_cmd index_weight_file model_bin_directory")
        exit(1)

    command_prefix = sys.argv[1]
    index_weight_file = sys.argv[2]
    model_bin_directory = sys.argv[3]
    
    # get weight matrixes from index file
    pre_datas = []
    with open(index_weight_file, 'r') as f:
        pre_datas = f.readlines()
    
    m_values = [1]
    iters = 1000
    
    for predata in pre_datas:
        datas = predata.rstrip()  # remove newline character
        datas = datas.split(" ")
        weight_file_name = datas[0]
        n = datas[1]
        k = datas[2]

        print("----------------> File: %s" % weight_file_name)
        filepath = model_bin_directory + '/'  + weight_file_name
        print("\t\tPerformence(GFlop/s)\tTime(ms)\tSize(Ops)")
        try:
            for m in m_values:
                print("< m = %d, k = %d, n = %d >" % (m, int(k), int(n)))
                command = command_prefix + " " + filepath + " " + str(m) + " " + \
                    k + " " + n + " " + str(iters)
                output = get_output(command)
                #print("CuSparse:\t%f\t\t%f\t%d" % tuple(output[:3]))
                #print("CuBlas:\t\t%f\t\t%f\t%d" % tuple(output[3:]))
                print("CuSparse: %s" % output[0])
                print("PaiSparse: %s" % output[1])
                print("CuBLAS: %s" % output[2])
        except IndexError:
            print("Computing matrix from %s fail..." % weight_file_name)
            continue
        
