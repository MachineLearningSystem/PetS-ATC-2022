import re
import os
import sys
import torch
import numpy as np

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


def model_parser(paths):
    model_bin_file = paths[0]
    index_file_path = paths[1]
    bin_dir = paths[2]
    
    print("Index file path: ", index_file_path)
    pattern = re.compile(r'(layer)(?!(.*LayerNorm).).*(weight)')
    # model = torch.load(model_bin_file, map_location='cpu')['module']
    model = torch.load(model_bin_file, map_location='cpu')

    # remove existed file to avoid write double time
    if os.path.exists(index_file_path):
        os.remove(index_file_path)
   
    for key, _ in model.items():
        if not re.search(pattern, key):
            continue
        if key.find('layernorm') > -1:
            continue
        print(key)
        # continue
        saved_file = key + '.bin'
        shape = model[key].cpu().numpy().astype(np.float16).shape
        write_str = saved_file + " " + str(shape[0]) + " " + str(shape[1])

        with open(index_file_path, 'a') as f:
            f.write(write_str + "\n")
        
        full_saved_path = bin_dir + '/' + saved_file
        model[key].cpu().numpy().astype(np.float16).tofile(full_saved_path)

if __name__ == '__main__':
    # config file parse
    if len(sys.argv) != 2:
        print("Usage: python model_parser.py config_file_path")
        exit(1)
    config_path = sys.argv[1]
    paths = config_parser(config_path)

    # generate model encoder|decoder layer weight matrix
    model_parser(paths)






    
