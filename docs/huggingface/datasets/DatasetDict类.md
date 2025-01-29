# DatasetDict 类

`datasets.DatasetDict` 是Hugging Face的 `datasets` 库中的一个类，它是一个包含多个 `datasets.Dataset` 对象的字典，通常用来存储和管理具有不同数据分割（如训练集、验证集和测试集）的数据集。

## 创建 `DatasetDict`

`DatasetDict` 可以从多个 `datasets.Dataset` 对象创建，或者从一个包含数据集的字典中创建，本质上 `DatasetDict` 是一个字典，其键是数据集的名称（如 "`train`", "`test`"），值是对应的 `datasets.Dataset` 对象。其构造函数如下：

``` python
class DatasetDict(dict):
    """A dictionary (dict of str: datasets.Dataset) with dataset transforms methods (map, filter, etc.)"""
    pass
```

`DatasetDict` 重要接口如下所示：

| 方法名 | 描述 | 备注 |
| ---- | ---- | ---- |
| from_json | 从JSON文件加载`DatasetDict`。| 读取数据之后，会存储多个关键的信息，这些信息如下：<br> 1.  cached_files: 每个集合的缓存文件 <br> 2. data: 加载之后的数据，这个数据是一个字典，字典的key是 `train`, `dev`, `test`。字典的value是对应的 `MemoryMappedTable` 对象。<br> 3. num_columns: 数据集合每个样本字段的数目。<br> 4. num_rows: 这是一个字段，保存了每个数据集的样本的数目，key是 `train`, `dev`, `test`。<br> 5. 接下来就是各种集合对应的数据，例如 `train`, `dev`, `test` 对应的数据。|

## `map` 方法
用法示例：
``` python
encoded_dataset = dataset.map(encode_examples, batched=True)
``` 
batched参数的作用
当使用DatasetDict的map方法时，batched参数控制着传递给你提供的函数的数据是逐条还是以批处理的形式。具体来说：

当batched=False（默认设置）时，提供的函数将被逐条应用于数据集中的每个元素。这意味着函数每次接收并处理数据集中的单条数据；
当batched=True时，提供的函数将以批处理的方式应用于数据集。这意味着函数一次性接收一批数据，这批数据的大小由batch_size参数（如果有提供的话）决定。这可以显著提高数据处理的效率，特别是当你的处理函数能够利用向量化操作时；
使用batched=True有几个潜在的好处：

效率提升：批处理可以减少函数调用的次数，并允许你在处理数据时利用更高效的向量化操作，从而提高整体处理速度；
内存管理：当处理非常大的数据集时，逐条处理可能会导致内存使用增加，特别是如果每次函数调用都需要加载额外的资源或模型。通过批处理，你可以更有效地管理内存使用；
并行处理：某些情况下，batched处理可以与num_proc参数结合使用，后者允许在多个工作进程上并行应用函数，进一步提高处理速度；
真的是批量处理吗
当你设置batched=True时，的确是以批次形式处理数据，但处理的具体方式取决于你的函数实现。在这种模式下，传递给你的函数的是一整批数据，而不是单条数据。这意味着你的函数一次接收到的是多条记录的集合（通常是一个字典，其中每个键对应的值是包含多个记录的列表），而函数内部如何处理这批数据，则取决于你的实现。

逐条执行：即使在batched=True的情况下，你的函数内部依然可以选择逐条处理批次中的每条数据。这种情况下，虽然数据是以批次形式传递给函数的，但如果你在函数内部遍历每条数据进行处理（如使用循环），实际上还是逐条执行。这种方法并没有充分利用batched=True带来的批处理优势，除非你的处理逻辑无法向量化；
批量执行：更高效的方法是使用向量化操作处理整批数据，这通常意味着利用NumPy、Pandas或其他支持批量操作的库来一次性处理整批数据。这种方式可以显著提高效率，因为它减少了Python循环的开销，同时利用了底层库的优化（如并行化处理）；
假设你想将文本数据转换为小写，以下两种方式都可以实现，但效率有所不同：

## 方法1：逐条处理（即使在batched=True的情况下）
def to_lower(batch):
    # 这里仍然是逐条处理，尽管数据是以批次形式传递的
    return {"text": [text.lower() for text in batch["text"]]}
## 方法2：利用向量化操作批量处理（如果可能）
def to_lower(batch):
    # 假设batch["text"]是一个Pandas Series或类似的结构，可以直接使用向量化的str.lower()方法
    return {"text": batch["text"].str.lower()}
总的来说，虽然batched=True允许你以批次的方式处理数据，但如何实现批量处理的效率优化，取决于你的函数实现和数据处理的具体方法。向量化操作通常提供了更好的性能。


## 参考：
1. http://118.195.180.129:5000/project-29/doc-153/
2. https://www.cnblogs.com/zhangxuegold/p/17531896.html
3. https://study.hycbook.com/article/57912