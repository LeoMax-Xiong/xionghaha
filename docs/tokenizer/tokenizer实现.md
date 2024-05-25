
# Tokenizer 实现

tokenizer负责准备输入以供模型使用。huggingface 的 `Tokenizer`库包含所有模型的tokenizer。大多数tokenizer都有两种版本：一个是完全的 Python 实现，另一个是基于 Rust 库 🤗 Tokenizers 的“Fast”实现。“Fast” 实现允许：

* 在批量分词时显著提速
* 在原始字符串（字符和单词）和token空间之间进行映射的其他方法（例如，获取包含给定字符的token的索引或与给定token对应的字符范围）。
  
基类 [`PreTrainedTokenizer`] 和 [`PreTrainedTokenizerFast`] 实现了在模型输入中编码字符串输入的常用方法（见下文），并从本地文件或目录或从库提供的预训练的 `tokenizer` 实例化/保存 python 和“Fast” tokenizer。它们都依赖于包含常用方法的 `PreTrainedTokenizerBase` 和 `SpecialTokensMixin`。

因此，`PreTrainedTokenizer` 和 `PreTrainedTokenizerFast` 实现了使用所有tokenizers的主要方法：

* 分词（将字符串拆分为子词标记字符串），将tokens字符串转换为id并转换回来，以及编码/解码（即标记化并转换为整数）。
* 以独立于底层结构（BPE、SentencePiece .....）的方式向词汇表中添加新 tokens。
* 管理特殊tokens（如mask、句首等）：添加它们，将它们分配给tokenizer中的属性以便于访问，并确保它们在标记过程中不会被分割。

BatchEncoding 包含 PreTrainedTokenizerBase 的编码方法（`__call__`、`encode_plus` 和 `batch_encode_plus`）的输出，并且是从 Python 字典派生的。当tokenizer是纯 Python tokenizer时，此类的行为就像标准的 Python 字典一样，并保存这些方法计算的各种模型输入（`input_ids`、`attention_mask` 等）。当分词器是“Fast”分词器时（即由 HuggingFace 的 tokenizers 库支持），此类还提供了几种高级对齐方法，可用于在原始字符串（字符和单词）与token空间之间进行映射（例如，获取包含给定字符的token的索引或与给定token对应的字符范围）。


## 参考
1. https://huggingface.co/docs/transformers/v4.41.1/zh/main_classes/tokenizer