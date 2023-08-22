> tensorflow/python/ops/embedding_ops.py
# embedding_lookup_sparse
``
@tf_export("nn.embedding_lookup_sparse")  
def embedding_lookup_sparse(params,  
                            sp_ids,  
                            sp_weights,  
                            partition_strategy="mod",  
                            name=None,  
                            combiner=None,  
                            max_norm=None)
``
- params：single tensor or a list of tensors

- sp_ids：N x M `SparseTensor` of int64 ids where N is typically batch size and M is arbitrary.

- sp_weights：either a `SparseTensor` of float / double weights, or `None` to indicate all weights should be taken to be 1. If specified, `sp_weights` must have exactly the same shape and indices as `sp_ids`.

- partition_strategy：if `len(params) > 1`. Currently `"div"` and `"mod"` are supported。
  - "mod"
    - id%len(params): index of the param in params
    - id//len(params): index of the element in the above param
    > note：embedding 求余切割成 len(params) 份
  - "div"
    - assign ids to partitions in a contiguous manner
    > note: embedding 按第一维按顺序切割成 len(params) 份

- name：name for the op

- combiner：support "sum" "mean" "sqrtn" 
  - "sum"：computes the weighted sum of the embedding results for each row
  - "mean"：weighted sum divided by the total weight(sp_weights)
  - "sqrtn"：weighted sum divided by the square root of the sum of the squares of the weights
    - > 权重平方和，再开根号作为除数

- max_norm：before combining，each embedding is clipped if its l2-norm is larger than this value

## api 调用
- array_ops.unique：去重

- embedding_lookup -> _embedding_lookup_and_transform
  - The returned tensor has shape `shape(ids) + shape(params)[1:]

- math_ops.sparse_segment_*：按 combiner 方式聚合，返回 shape = `batch size`*`dim`
