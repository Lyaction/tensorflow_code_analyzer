# sparse_column_with_embedding
> contrib api
- 返回的对象及继承关系 
  - _SparseColumnEmbedding -> _SparseColumn -> (_FeatureColumn, fc_core._CategoricalColumn -> _FeatureColumn)
- 使用时内嵌入 _EmbeddingColumn，继承关系
  - _EmbeddingColumn -> (_FeatureColumn, fc_core._DenseColumn -> _FeatureColum)
# input_layer
``
def _internal_input_layer(features,
                          feature_columns,
                          weight_collections=None,
                          trainable=True,
                          cols_to_vars=None,
                          scope=None,
                          cols_to_output_tensors=None,
                          from_template=False)
``
1._normalize_feature_columns ，check 类型，转 list 对象；check name 重复
2.ops.GraphKeys.GLOBAL_VARIABLES 及 ops.GraphKeys.MODEL_VARIABLES 加入weight_collectionsoo
3.构建 _LazyBuilder(features)
4.tensor = column._get_dense_tensor(builder,weight_collections=weight_collections,trainable=trainable)
5.外部函数 _embeddings_from_arguments() 参数字典 _DeepEmbeddingLookupArguments
  - embeddings = variable_scope.get_embedding_variable_internal()
    - fixed_size_partitioner
  - 查询 embedding_ops.safe_embedding_lookup_sparse()
    - mode = div
