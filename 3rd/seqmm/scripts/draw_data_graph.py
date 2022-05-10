import sys
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
import numpy as np
import re

import torch

def config_parser(config_file):
    '''
    model_bin_file = paths[0]
    index_file_path = paths[1]
    bin_dir = paths[2]
    ...
    '''
    paths = []
    with open(config_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.find(":") >= 0:
                path = line.split(":")[1].strip()
                paths.append(path)
    return paths

def func(pct, allvals):
    # absolute is in half data type
    absolute = int(round(pct/100.*np.sum(allvals)) * 2 / (1000*1000*1000))
    return "{:.1f}%\n({:d} GB)".format(pct, absolute)

def drawpie(size_series, output):
    labels = size_series.index
    sizes = size_series.values
    explode = tuple([0] * len(labels))
    plt.switch_backend('agg')
    _, ax1 = plt.subplots(figsize=(16, 9)) # set pane size


    #_, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
    #                              autopct='%1.2f%%', shadow=False,
    #                              startangle=170)

    wedges, texts, autotexts = ax1.pie(
        sizes,
        explode=explode,
        autopct=lambda pct: func(pct, sizes), shadow=False,
        startangle=170)
    
    ax1.axis('equal')

    ax1.legend(wedges, labels)

    # set font size
    # font size include: ‘xx-small’,x-small’,
    # 'small’,'medium',‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    proptease = fm.FontProperties()
    proptease.set_size('medium')
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)

    plt.savefig(output, dpi=800)
    plt.show()


def get_encoder_decoder_mem_usage_breakdown(index_file_path):
    encoder_size, decoder_size = 0, 0
    encoder_weight_names = []
    decoder_weight_names = []
    encoder_weight_sizes = []
    decoder_weight_sizes = []
    with open(index_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            datas = line.split(" ")
            weight_name = datas[0]
            weight_size = int(datas[1]) * int(datas[2])
            if weight_name.find('encoder') > -1:
                encoder_weight_names.append(weight_name)
                encoder_weight_sizes.append(weight_size)
                encoder_size += weight_size
            else:
                decoder_weight_names.append(weight_name)
                decoder_weight_sizes.append(weight_size)
                decoder_size += weight_size
    result = {"encoder_size": encoder_size,
              "decoder_size": decoder_size,
              "encoder_weight_names": encoder_weight_names,
              "encoder_weight_sizes": encoder_weight_sizes,
              "decoder_weight_names": decoder_weight_names,
              "decoder_weight_sizes": decoder_weight_sizes}
    return result
    
def get_mem_usage_breakdown(model_file):
    encoder_weight_size_total, decoder_weight_size_total, others = 0, 0, 0

    model = torch.load(model_file, map_location='cpu')['module']

    for key, _ in model.items():
        size = model[key].cpu().numpy().astype(np.float32).size
        print("%s, %d" % (key, size))
        pattern = re.compile(r'(layer)(?!(.*LayerNorm).).*(weight)')
        if re.search(pattern, key):
            if key.find('encoder') > -1:
                encoder_weight_size_total += size
            else:
                decoder_weight_size_total += size
        else:
            others += size

    '''
    encoder_weight_size_total = 19327352832
    decoder_weight_size_total = 6442606592
    others = 506803203
    '''
    
    result = {"encoder_weight_size": encoder_weight_size_total,
              "decoder_weight_size": decoder_weight_size_total,
              "others_size": others}

    print(result)
    
    return result


def draw_matrix(matrix, ouput, dpi):
    plt.switch_backend('agg')
    plt.matshow(matrix)
    plt.savefig(ouput, dpi = dpi)


def draw_distribution_plane(bin_file, k, n, output, dpi):
    matrix = np.fromfile(bin_file, dtype=np.float16)
    matrix = np.int64(abs(matrix) == 0.0).reshape(k, n)
    #matrix = matrix[:768, :768]
    draw_matrix(matrix, output, dpi)

def draw_distribution_hist(bin_file, k, n, output_dir, title, dpi):
    plt.switch_backend('agg')
    ## Compute histogram information and do plot.
    matrix = np.fromfile(bin_file, dtype=np.float16)
    matrix = np.int64(abs(matrix) != 0.0).reshape(k, n)
    x = matrix.nonzero()
    print(x[0])
    print(x[1])
    row_dist = np.unique(x[0], return_counts=True)[1]
    col_dist = np.unique(x[1], return_counts=True)[1]

    bins = range(np.amax(row_dist) + 2)
    plt.hist(row_dist, bins, log=True)
    plt.title(title + " (row)")
    plt.savefig(output_dir+"/"+title+".hist.row.jpg", dpi = dpi)

    plt.clf()

    bins = range(np.amax(col_dist) + 2)
    plt.hist(col_dist, bins, log=True)
    plt.title(title + " (col)")
    plt.savefig(output_dir+"/"+title+".hist.col.jpg", dpi = dpi)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python draw_data_graph.py config_file_path")
        exit(1)
    config_file = sys.argv[1]
    paths = config_parser(config_file)
    index_file_path = paths[1]
    bin_path = paths[2]

    '''
    # get encoder|decoder|others size
    result = get_mem_usage_breakdown(paths[0])
    
    # draw encoder-decoder-others pie
    print("========= Draw encoder-decoder-others pie ===========")
    encoder_decoder_pie = paths[3]
    encoder_decoder_values = [result["encoder_weight_size"],
                              result["decoder_weight_size"],
                              result["others_size"]]
    encoder_decoder_index = ["encoder_weight_size", "decoder_weight_size",
                             "others_size"]
    encoder_decoder_series = pd.Series(encoder_decoder_values,
                                       index = encoder_decoder_index)
    drawpie(encoder_decoder_series, encoder_decoder_pie)

    # get encoder|decoder size
    result = get_encoder_decoder_mem_usage_breakdown(paths[1])
    
    # draw encoder-decoder pie
    print("========= Draw encoder-decoder pie ===========")
    encoder_decoder_pie = paths[4]
    encoder_decoder_values = [result["encoder_size"], result["decoder_size"]]
    encoder_decoder_index = ["encoder_size", "decoder_size"]
    encoder_decoder_series = pd.Series(encoder_decoder_values,
                                       index = encoder_decoder_index)
    drawpie(encoder_decoder_series, encoder_decoder_pie)

    # draw encoder layer0 pie
    print("========= Draw encoder layer0 pie ===========")
    encoder_layer0_pie = paths[5]
    encoder_layer0_index = result["encoder_weight_names"][:4]
    encoder_layer0_values = result["encoder_weight_sizes"][:4]
    encoder_layer0_series = pd.Series(encoder_layer0_values, index=encoder_layer0_index)
    drawpie(encoder_layer0_series, encoder_layer0_pie)

    # draw decoder layer0 pie
    print("========= Draw decoder layer0 pie ===========")
    decoder_layer0_pie = paths[6]
    decoder_layer0_index = result["decoder_weight_names"][:6]
    decoder_layer0_values = result["decoder_weight_sizes"][:6]
    decoder_layer0_series = pd.Series(decoder_layer0_values, index=decoder_layer0_index)
    drawpie(decoder_layer0_series, decoder_layer0_pie)
    '''

    '''
    # draw encoder layer0 matrix distribution
    print("========= Draw encoder layer0 matrix distribution ===========")
    output_dir = paths[7]
    encoder_layer0_bin_paths_and_shapes = []
    with open(index_file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.find("encoder.layer.0") > -1:
                encoder_layer0_bin_paths_and_shapes.append(line.split(" "))

    for matrix_data in encoder_layer0_bin_paths_and_shapes:
        print("---> File: %s , k * n: %d " % (matrix_data[0], int(matrix_data[1]) * int(matrix_data[2])))
        output_file = output_dir + "/" + matrix_data[0] + ".jpg"
        bin_file = bin_path + "/" + matrix_data[0]
        draw_distribution(bin_file, int(matrix_data[1]), int(matrix_data[2]),
                          output_file, dpi=800)
    '''

    print("========= Draw decoder layer0 matrix distribution ===========")
    output_dir = paths[8]
    decoder_layer0_bin_paths_and_shapes = []
    with open(index_file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.find("decoder.layer.3") > -1:
                decoder_layer0_bin_paths_and_shapes.append(line.split(" "))

    for matrix_data in decoder_layer0_bin_paths_and_shapes:
        print("---> File: %s , k * n: %d " % (matrix_data[0], int(matrix_data[1]) * int(matrix_data[2])))
        output_file = output_dir + "/" + matrix_data[0] + ".jpg"
        bin_file = bin_path + "/" + matrix_data[0]
        draw_distribution_plane(bin_file, int(matrix_data[1]), int(matrix_data[2]),
                                output_file, dpi=800)
       # draw_distribution_hist(bin_file, int(matrix_data[1]), int(matrix_data[2]),
       #                        output_dir, matrix_data[0], dpi=800)
        
    

