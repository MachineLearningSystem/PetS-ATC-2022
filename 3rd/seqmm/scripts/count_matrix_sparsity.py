import os
import numpy as np

def count_sparsity(filepath):
    matrix = np.fromfile(filepath, dtype=np.float32)
    size = matrix.size
    count = np.count_nonzero(matrix == 0)

    # for v in matrix:
    #     if v - 0.0 < 1.0e-32:
    #         count += 1
    return (size, count / size)


if __name__ == '__main__':
    model_bin_directory = "/home/xuechao.wxc/model/bin/"
    files = os.listdir(model_bin_directory)
    files.sort()
    for file in files:
        print("----------------> File: %s" % file)
        filepath = model_bin_directory + '/'  + file
        pre_data = count_sparsity(filepath)
        print("size: %d\tsparsity: %f" % pre_data)
        # print("k: %d\tn: %d\tSize: %d\tcount: %d\tReal size: %d\tSparsity: %f\n" % tuple(pre_data))

