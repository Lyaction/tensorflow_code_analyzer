# Feature Column 解读
以 sparse_column_with_embedding 为例逐步展开介绍

---

## sparse_column_with_embedding
> contrib api
- 返回的对象及继承关系 
  - _SparseColumnEmbedding -> _SparseColumn -> (_FeatureColumn, fc_core._CategoricalColumn -> _FeatureColumn)
- 使用时内嵌入 _EmbeddingColumn，继承关系
  - _EmbeddingColumn -> (_FeatureColumn, fc_core._DenseColumn -> _FeatureColum)
### input_layer
1._normalize_feature_columns ，check 类型，转 list 对象；check name 重复
