# Benchmarks

## Prune

### Run
Run prune example and measure performance.

```bash
~> cd build/benchmarks
~> ./plug_prune_example weight_file_path m k n threshold iter
```

Model weight matrices can be download from the [link](https://file.alibaba-inc.com/library/1dece649-c8b2-4ba3-929b-691e6b9b1234/%E7%A7%81%E4%BA%BA%E6%96%87%E4%BB%B6%E5%BA%93/Sparse-PLUG/tap/weight).

Together with the weight matrices, there also contains an index file named `weight_index.txt` in the same [link](https://file.alibaba-inc.com/library/1dece649-c8b2-4ba3-929b-691e6b9b1234/%E7%A7%81%E4%BA%BA%E6%96%87%E4%BB%B6%E5%BA%93/Sparse-PLUG/tap/weight). It contains the matrics names as well as their shapes. For example, an entry looks like:

```
decoder.decoder.layer.0.attention.query_key_value.weight.bin 24576 8192
```

The example will save the pruned matrix in a file named `pruned_weight_matrix.bin`.

### Check
Check pruned matrix non-zero elements distribution.

```bash
~> cd scripts
~> python draw_matrix_distribution.py pruned_weight_matrix_path row col
```

3 jpg figures will be generated, representing non-zero elements distributions according to rows (`hist.row.jpg`) and columns (`hist.col.jpg`), and visualized non-zero elements in the matrix plane (`plane.jpg`).