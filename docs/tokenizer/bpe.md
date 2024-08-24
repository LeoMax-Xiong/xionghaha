# BPE 算法

BPE 算法的全称是 Byte-Pair Encoding, 最早是 1994 年提出应用于 [数据压缩](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) 任务。在 2015 年, [Neural Machine Translation of Rare Words with Subword Units]((https://arxiv.org/abs/1508.07909)) 提出将其延申至 **文本分词** 任务, 同时没有改变算法的名称。(**如果你不了解之前的数据压缩算法, 可能会好奇为什么是这个名字**)。

OpenAI 关于 GPT 的一系列工作, 以及 Meta 关于 LLaMA 的一系列工作都是使用这种分词方式。OpenAI 虽然没有开源 GPT-3 模型的参数权重, 但是开源了分词库 [tiktoken](https://github.com/openai/tiktoken), 帮助用户在调用接口前计算 token 数。

下面就让我们看看这个算法。首先来看看 "训练" 过程:

* 第一步：使用 **空格** 对语料库中的所有句子进行拆分，得到所有可能的 `word`。

* 第二步：将语料库中所有的 `word` 以字符 (最小颗粒) 为单位进行拆分, 每一个字符作为一个 subword。比方说, "`best`" 被拆分成 "`b e s t`" 四个 subword (用 **空格** 作为分隔符)。

* 第三步, 初始化 **词表** 和 **合并规则列表**。其中, **词表** 就是所有可能的 **subword** 集合, 也就是初始化成语料库中所有可能的字符集合。

* 第四步, 遍历 `word` 中所有可能的 **subword pair**, 统计他们出现的频数。然后将数据集中 频数最高 的那一个 subword pair 合并在一起, 合并后的 subword 加入 **词表** 中, **合并规则加入** 合并规则列表 中。

    举例来说, "`b e s t`" 现在有三种可能的 **subword pair**: "`b e`", "`e s`" 和 "`s t`"。统计完成后, 发现在语料库中, "`s t`" 出现的频数最高。那么, 我们就将整个数据集中的 "`s t`" 合并起来, 变成 "`st`"。此时 `best` 的拆分结果就是 "`b e st`", 从四个 subword 变成三个 subword。

    同时, 将 "`st`" 这个 subword 加入 **词表** 中, 将 ("`s t`", "`st`") 加入 **合并规则列表** 中。

* 第五步, 一致重复 第三步 的操作, 直至 **词表** 达到预设的大小即可。

整个训练过程就是不断地去寻找语料库中频数最高的 **subword pair**, 然后合并成一个 **subword**, 和 [数据压缩](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) 算法中的核心思想是一致的。

需要注意的是, **subword pair** 的合并次数并不一定等于其频数。举例来说, 对于 "`x x x`" 来说, 其中 "`x x`" 出现了两次, 但是我们合并时只能合并一个, 即 "`xx x`"。

那么, 我们怎么对 word 进行分词呢? 首先, 将 word 以字符为单位进行拆分, 一个字符作为一个 subword。然后按照 **合并规则列表** 中的顺序, 如果出现了 **subword pair**, 就进行合并。最后剩下的 subword 列表就是最终的分词结果。

整体上的方案就是这样, 可以说非常巧妙。这个算法开源在 [subword-nmt](https://github.com/rsennrich/subword-nmt) 项目中。

现在, 很多论文中都说其使用了 Byte-Level BPE (BBPE), 就是将第四部分所说的 byte-level 方案和 BPE 结合起来, 最初是 OpenAI 在 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 工作中提出。网上很多博客在介绍 BBPE 时, 引用的论文是 Meta 的 [Neural Machine Translation with Byte-Level Subwords](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.03341), 其发布比 GPT-2 晚半年, 不要弄错关系了。


## 训练
### 第一步：使用空格对语料库中的所有句子进行拆分

首先, 我们需要一个训练语料库，使用 **空格进行分词**，在 subword-nmt 中实现的代码如下：
``` python
# 统计语料库中所有单词出现的频数
for i, line in enumerate(fobj):
    # 使用空格分割
    for word in line.strip('\r\n ').split(' '):
        if word:
            vocab[word] += 1
```

### 第二步：将 word 以字符为单位进行拆分

将语料库中所有的 `word` 以字符 (最小颗粒) 为单位进行拆分，每一个字符作为一个 subword，并且在最后一个字符后面添加一个 `</w>` 形成新的字符。比方说, "`best`" 被拆分成 "`b e s t</w>"` 四个 subword (用 **空格** 作为分隔符)。

在 subword-nmt 中实现的代码如下：

``` python
vocab = dict([(tuple(x[:-1]) + (x[-1] + '</w>',) ,y) for (x, y) in vocab.items()])
```

### 第三步：根据字符的频数进行降序排序

获取字符的频率之后，根据字符的序列进行降序排序，得到字符的排序序列。
``` python
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
```


### 第四步：统计所有单词中bi-gram的频数

在训练语料的所有单词中，统计所有单词中bi-gram的频数，具体的实现代码如下：

``` python
def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices
```

### 第五步：合并频率最高的bi-gram

在第 4 步中，统计了所有单词中bi-gram的频数，然后根据频率进行降序排序，取出频率最高的bi-gram，合并成新的字符，将训练语料中所有出现该bi-gram都进行合并：
``` python
def replace_pair(pair, vocab, indices):
    """
    将出现的所有的('A', 'B')替换成'AB'。
    Args:
        pair: 一个由两个元素组成的元组，代表待替换的符号对，形如 ('A', 'B')。
        vocab: 一个由词及其词频组成的列表，其中每个元素都是一个元组，
               形如 (word, freq)，其中 word 是一个由单词组成的元组，freq 是该词的词频。
        indices: 一个字典，用于存储符号对在 vocab 中的索引。
    
    Returns:
        一个列表，包含替换过程中发生的变化，每个元素都是一个四元组，形如 (j, new_word, word, freq)，
        其中 j 是替换后的词在 vocab 中的索引，new_word 是替换后的词，word 是替换前的词，freq 是该词的词频。
    
    """
    first, second = pair
    pair_str = ''.join(pair)    # 将广义的bi-graim合并在一起，形如 ('A', 'B') -> 'AB'
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    # 这个正则表达式用于匹配字符串中由 first 和 second 组成的，前后不接触其他非空白字符的模式。
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        # 获取bi-gram在indices字典中的索引
        # bi-gram的索引，表示的是该字符对出现的位置以及该位置上出现的频率
        iterator = indices[pair].items()
    
    # j 表示pari出现的位置，freq 表示pair在该位置上出现的频率
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes
```

### 第六步：重新计算bi-gram的频率

在第五步中，将所有的bi-gram都合并成新的字符，那么就需要重新计算每个字符的频率，计算的代码如下所示：

``` python
def update_pair_statistics(pair, changed, stats, indices):
    """
    更新符号对的索引和频率
    如果合并了符号，那么只需要更新所有包含该符号的符号对。
    
    Args:
        pair (tuple): 需要合并的符号对
        changed (list): 包含发生变化的四元组 (j, word, old_word, freq)，
                        其中 j 为变化位置，word 为变化后的单词，old_word 为变化前的单词，freq 为变化频率
        stats (dict): 符号对的频率统计
        indices (dict): 符号对的索引统计
    
    Returns:
        None
    
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                # 在老的 word 中找第一个 first 出现的位置
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            # 如果找到对应的bi-gram, 更新频率和索引
            if i < len(old_word) - 1 and old_word[i + 1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:   # 表示i不是0，i-1>=0，假设是原始的序列可能是('A', 'B', 'C'), ('B', 'C') 是要合并的序列
                    prev = old_word[i-1:i+1]    # 取出 ('A', 'B')，因为('B', 'C')要合并了，因此不存在('A', 'B') 组合，会产生新的('A', 'BC')
                    stats[prev] -= freq         # 统计信息中去掉当前 word 中 ('A', 'B') 的频率
                    indices[prev][j] -= 1       # 索引中去掉当前 word 中 ('A', 'B') 的索引
                if i < len(old_word) - 2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    # 1. 不是 (A, B, C, B) 序列
                    # 2. 如果是 (A, B, C, B) 序列，那么当前的(B, C, B) 是整个序列的最后三个字符
                    # 3. 如果是 (A, B, C, B) 序列，那么当前的(B, C, B) 不是整个序列的最后三个字符，那么不能是(A, B, C, B, C) 这种情形
                    if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                        # 获取下一个bi-gram，假设当前的序列是 ('A', 'B', 'C', 'D')，
                        # 由于合并了 ('B', 'C')，那么下一个bi-gram是 ('C', 'D')，该句子上当前位置一定不会再出现('C', 'D')了
                        # 这是因为当前的('C') 被('B', 'C') 合并了，应该出现的是('A', 'BC') 与 ('BC', 'D')
                        nex = old_word[i + 1:i + 3] # 取出 ('C', 'D')
                        stats[nex] -= freq          # 移除当前句子中该位置贡献的 ('C', 'D') 的频率
                        indices[nex][j] -= 1
                i += 2  # 跳过 ('B', 'C')
            else:
                # 可能出现 ('B', 'B', 'C')  这种情况，找到第一个 ('B') 之后并不满足条件，跳过，继续找下一个 ('B')
                # 直到找到 ('B', 'C') 满足条件，进入到上面的 if 块中
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                # 在新的 word 中找第一个 new_pair 出现的位置
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:   # 新的符号不是在句首
                prev = word[i - 1:i + 1]    # 取出 ('A', 'BC')
                stats[prev] += freq         # 统计信息中增加当前 word 中 ('A', 'BC') 的频率
                indices[prev][j] += 1       # 索引中增加当前 word 中 ('A', 'BC') 的索引
            
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            # 如果合并了 ('B', 'C')，那么当前位置的下一个字符是 'D'，那么 ('BC', 'D') 应该被增加到词典中
            # 如果当前位置的下一个字符是 'BC'，那么 ('BC', 'BC') 不处理
            # 这是因为在找下一个 ('BC')的时候，会通过统计 ('BC', 'BC') 作为 prev 添加到统计信息中，
            # 如果通过 nex 添加，那么下面通过 prev 会重复计算
            if i < len(word) - 1 and word[i + 1] != new_pair:   
                nex = word[i:i + 2]     # 取出 ('BC', 'D')
                stats[nex] += freq      # 统计信息中增加当前 word 中 ('BC', 'D') 的频率
                indices[nex][j] += 1    # 索引中增加当前 word 中 ('BC', 'D') 的索引
            i += 1

```

### 第七步：剪枝

在词典中去掉不符合要求的词，比如频率过低的词，实现的代码如下：
```python
def prune_stats(stats, big_stats, threshold):
    """
    根据阈值修剪统计字典，以提高max()操作的效率。
    为了优化max()的效率，修剪统计字典

    符号对的频率永远不会增加，因此修剪通常是安全的
    （直到我们发现最频繁的符号对比之前修剪过的符号对的频率还要低）
    big_stats保留了完整的统计信息，以便在我们需要访问被修剪的项时使用
    
    Args:
        stats (dict): 符号对统计字典。
        big_stats (dict): 完整的符号对统计字典，用于访问已修剪的项。
        threshold (int): 修剪阈值，小于该值的符号对将被修剪。
    
    Returns:
        None
    
    注意：
        符号对的频率永远不会增加，因此修剪通常是安全的（直到最频繁的符号对的频率低于之前修剪的符号对的频率）。
        big_stats保存完整的统计信息，以便在需要时访问已修剪的项。
    """
    for item, freq in list(stats.items()):
        # 删除频率小于阈值的项
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq

```


## 推理

