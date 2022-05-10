# Scripts Usage

Scripts list:

- model_parser.py
- draw_data_graph.py
- count_matrix_sparsity.py
- test_spmm_and_gemm.py

## model_parser.py

解析Pytorch模型，得到相应的权重矩阵并将其存储至指定路径。模型、生成的权重矩阵以及对应索引文件的路径均由配置文件指出。

### 配置文件示例

```ini
[ Input ]
Model binary file: ~/model/mp_rank_00_model_states/mp_rank_00_model_states.pt

[ Output ]
Index file: ~/model/mp_rank_00_model_states/weight_index.txt
Binary file directory: ~/model/mp_rank_00_model_states/bin
Encoder-Decoder pie picture: ~/model/mp_rank_00_model_states/encoder_decoder_pie.jpg
Encoder layer0 pie picture: ~/model/mp_rank_00_model_states/encoder_layer0_pie.jpg
Decoder layer0 pie picture: ~/model/mp_rank_00_model_states/decoder_layer0_pie.jpg
Encoder layer0 distribution picture directory: ~/model/mp_rank_00_model_states/encoder_layer0_distribuitions
```

### 使用说明

```bash
python model_parser.py config_file
```

## draw_data_graph.py

draw_data_graph.py 为矩阵生成Encoder与Decoder比例图（Encoder-Decoder pie picture）、Encoder layer0 与Decoder layer0矩阵的比例图以及Encoder layer0中矩阵的分布图。

使用说明：

```bash
python draw_data_graph.py config_file
```

## count_matrix_sparsity.py

根据 `model_bin_directory`，统计其中所有矩阵的稀疏度，并打印输出。

```bash
python count_matrix_sparsity.py
```

## test_spmm_and_gemm.py

test_spmm_and_gemm.py 会根据`model_bin_directory` 和`index_weight_file`，统计其中每个矩阵的Spmm(CuSparse)与Gemm(CuBlas)的运行时间。计算过程中，矩阵A的Row的值m取值范围是[1,8,16,32,64,128]。

```bash
python test_spmm_and_gemm.py
```

