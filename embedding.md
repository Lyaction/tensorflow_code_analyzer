# Feature Column 解读
<p>以 sparse_column_with_embedding 为例逐步展开介绍</p>

---

## sparse_column_with_embedding
<p>
contrib api<br>
返回的对象及继承关系 _SparseColumnEmbedding -> _SparseColumn -> (_FeatureColumn, fc_core._CategoricalColumn -> _FeatureColumn)<br>
使用时内嵌入 <strong> _EmbeddingColumn</strong>，继承关系为 _EmbeddingColumn -> (_FeatureColumn, fc_core._DenseColumn -> _FeatureColum)
</p>
### input_layer
<ol>
<li>_normalize_feature_columns ，check 类型，转 list 对象；check name 重复</li>
<li></li>
<li></li>
<li></li>
</ol>
