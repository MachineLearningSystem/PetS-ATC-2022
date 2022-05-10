import sys
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
import numpy as np
import scipy.sparse as sparse

def draw_matrix(matrix, ouput, dpi):
    plt.switch_backend('agg')
    #plt.matshow(matrix)
    plt.spy(matrix, marker='o', markersize=1.0, markeredgewidth=0)
    plt.savefig(ouput, dpi = dpi)

def draw_distribution_plane(bin_file, m, n, title, dpi):
    matrix = np.fromfile(bin_file, dtype=np.float16)
    matrix = np.int64(abs(matrix) > 0.0).reshape(m, n)
    matrix = matrix[:1024, :1024]
    draw_matrix(matrix, title+".jpg", dpi)

def draw_distribution_hist(bin_file, m, n, title, dpi):
    plt.switch_backend('agg')
    ## Compute histogram information and do plot.
    matrix = np.fromfile(bin_file, dtype=np.float16)
    matrix = np.int64(abs(matrix) != 0.0).reshape(m, n)
    x = matrix.nonzero()
    print(x[0])
    print(x[1])
    print(len(x[0]))
    row_dist = np.unique(x[0], return_counts=True)[1]
    col_dist = np.unique(x[1], return_counts=True)[1]

    bins = range(np.amax(row_dist) + 2)
    plt.hist(row_dist, bins, log=True)
    plt.title(title + " (row)")
    plt.savefig(title+".row.jpg", dpi=dpi)

    # Clear frame.
    plt.clf()

    bins = range(np.amax(col_dist) + 2)
    plt.hist(col_dist, bins, log=True)
    plt.title(title + " (col)")
    plt.savefig(title+".col.jpg", dpi=dpi)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python " + sys.argv[0] + " matrix_file_path m n")
        exit(1)
    matrix_file = sys.argv[1]
    m = int(sys.argv[2])
    n = int(sys.argv[3])

    print("========= Draw matrix distribution ===========")
    print("---> File: %s , m * n: %d " % (matrix_file, m * n))
    draw_distribution_plane(matrix_file, m, n, "plane", dpi=1600)
    #draw_distribution_hist(matrix_file, m, n, "hist", dpi=800)
