# RedPajama：用于训练大型语言模型的开放数据集
> Maurice Weber, Daniel Y. Fu, Quentin Anthony Yonatan Oren Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun1, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, Ce Zhang

## 摘要
大型语言模型正日益成为人工智能、科学和整个社会中一项基石技术，然而，数据集构成和过滤的最优策略在很大程度上仍然难以捉摸。许多顶级性能的模型，其数据集管理和模型开发过程中仍然缺乏透明度，这阻碍了完全开源语言模型的发展。 在本文中，我们确定了必须解决的与数据相关三个核心挑战，以推动完全开源语言模型的发展。 这些包括：（1）模型开发的透明度，包括数据管理过程；（2）获取大量高质量数据；以及（3）数据集获取所需要的管理和分析组件和元数据。 为了应对这些挑战，我们发布了 RedPajama-V1，这是 LLaMA 训练数据集的开源复制版本。此外，我们还发布了 RedPajama-V2，这是一个庞大的只有网络数据的数据集，包含原始的、未经过滤的文本数据以及质量信号和元数据。RedPajama 数据集总共包含超过 100 万亿个符元，涵盖多个领域，其质量信号有助于数据过滤，旨在激发众多新数据集的开发。到目前为止，这些数据集已被用于训练生产上使用的强大语言模型，例如 Snowflake Arctic、Salesforce 的 XGen 和 AI2 的 OLMo。为了深入了解 RedPajama 的质量，我们使用参数最多为 16 亿的仅解码器语言模型进行了一系列分析和消融研究。我们的研究结果表明，如何有效地利用网络数据的质量信号来整理数据集的高质量子集，突出了 RedPajama 在大规模推进透明和高性能语言模型发展的潜力。

## 1.引言

预训练数据是现代大型语言模型 (LLM) 开发中最核心的组成部分之一。然而，该领域面临的核心挑战之一是预训练数据构成和管理策略普遍缺乏透明度[8]。事实上，除了一些值得注意的例外情况[19, 4, 2, 65]，大多数最先进 LLM 的报告文档[1]提供的预训练数据集细节很少，甚至没有。即使是像LLaMA[57, 58]这样的开源权重模型，也几乎没有提供关于其训练数据的细节，更不用说发布其数据集了。此外，研究和构建最佳数据组合，以及开发过滤规则和启发式方法的过程非常耗时，因为它需要对训练数据的不同组合进行多次消融实验。为了应对这些挑战，并以实现开源大语言模型访问和开发民主化的总体目标，我们发布了RedPajama数据集，该数据集总共包含超过100万亿个文本数据符元。 本着这一目标，我们使用以下设计原则来指导我们创建开放数据集的方法：

* **透明性**。我们至少从两个角度力求最大限度的**透明性**。 一方面，这意味着要记录并公开所有数据整理方面的信息。另一方面，我们努力构建开源透明的数据集，这使得应用程序开发者和研究人员能够更好地理解和设计语言模型。

* **规模**。庞大的可访问数据池是功能最强大的大型语言模型[52, 11, 58, 1]的核心组成部分之一，但由于构建、整理和存储它们需要大量的资源和专业知识，因此难以获得。因此，除了透明性之外，我们还力求规模化。

* **通用性**。我们的目标是通过提供通用的资源，为社区提供构建最先进的开源语言模型所需的数据集和组件。我们不规定高质量数据集的构成，而是提供一个广泛的、通用的网络文档语料库。每个文档都带有质量信号标签，使用户能够根据其特定需求和标准做出明智的决策。

![第一张图片](./img/1.png)
<!-- <figure>
<img src="./img/1.png">
<figcaption>图1:RedPajama数据集生态系统。 RedPajama为多个开源大语言模型（LLM）提供了预训练数据，包括OpenELM[36]、OLMo[19]、Snowflake的Arctic[54]和RedPajama-INCITE。 SlimPajama是RedPajama-V1的清理和去重版本。</figcaption>
</figure> -->

遵循这些原则，我们开发并公开发布了用于LLM预训练的 RedPajama 数据集。RedPajama-V1是一个公开可用、完全开源的、尽力而为的复现[57]中描述的数据集，用于训练第一版本迭代 LLaMA 系列模型。与该数据集一起，我们开发了 RedPajama-INCITE 模型，其中包括3B和7B规模的基础模型、指令微调模型和对话模型。基于这些努力的第一版经验教训，我们构建了 RedPajama-V2 数据集。 这个最新的数据集采用了完全不同的方法，并专门关注网络数据。它包含五种语言，来源是84个Common Crawl快照，时间范围从2014年到2023年。除了原始文本数据外，我们还在语料库的50T符元子集中发布了伴随每个文档的质量信号，目标是促进对过滤网络数据方法的原则性理解的研究。此外，在这项工作中，我们还提出了一系列消融研究，展示了质量信号对于创建不同质量的原始语料库子集以及后续模型性能的价值。

总而言之，在这项工作中，我们做出了以下贡献：

* **C1** 我们发布了 RedPajama-V1 数据集，这是用于训练LLaMA-1[57]的开源复现数据集。我们还包括一份关于创建语料库的考虑因素的详细报告。
* **C2** 我们报告了 RedPajama-INCITE 模型训练的过程和评估方法，包括我们如何使用Summit超级计算机以及我们必须应对的挑战的细节。
* **C3** 我们发布了 RedPajama-V2 数据集，这是最大的开源预训练数据集，它包含从网络上抓取的原始、未过滤的数据，以及为每个文档计算的 46 项质量指标，以促进数据整理方面的进一步研究。
* **C4** 基于 RedPajama-V2 数据集，我们对具有 4.68 亿参数的仅解码器 Transformer 模型进行了一系列消融研究，展示了如何利用质量信号创建在常见 NLP 基准测试中性能不同的模型。

本文的其余部分组织如下。在第 2 节中，我们将 RedPajama 数据集置于当前开源预训练数据集的格局中。第 3 节描述了 RedPajama-V1 背后的数据集创建过程的细节，以及构建 RedPajama-INCITE 模型系列的过程。第 4 节继续介绍我们的仅限网络的数据集 RedPajama-V2。除了描述数据处理步骤外，我们还介绍了数据集统计数据和消融研究。最后，我们在第 5 节中总结。

## 2.相关工作

许多工作都集中在构建大型语言模型的预训练数据集上。一些数据集是从各种来源混合整理的，而另一些数据集则完全来自网络数据。在仅限网络的数据集中，C4 数据集 [46] 是首批大型网络数据集之一，它包含一个从 CommonCrawl 过滤出的 1750 亿个符元的网络语料库。C4 仍然是网络数据集质量的基准。最近，RefinedWeb [44] 和 FineWeb [43] 已经证明，仅限网络的数据可以在无需组合多个领域的情况下产生强大的模型，并且与我们的工作类似，也提供了关于其数据整理技术的充足细节。与这些数据集形成对比的是，RedPajama-V2由100万亿个原始的、大部分未经过滤的文本符元组成。RedPajama-V2拥有40多个用于潜在过滤的质量信号，它推广了一种完全不同的方法，旨在为未来的高质量网络数据集设定新的标准，为下一代高质量网络数据集提供强大的基础。此外，[45]中提出的Gopher规则也具有重要意义，这些规则一直是许多前面提到的开源预训练数据集的核心。

作为对仅限网络数据集的补充，复合数据集引入了额外的领域并实现了更广泛的覆盖范围。最值得注意的是，The Pile [17]是最早的完全开源数据集之一。在LLaMA [57]发布之后（LLaMA使用了七个单独的子集和多个领域），我们发布了RedPajama-V1，这是一个LLaMA方案的开源复现版本，并得到了广泛采用。在此基础上，SlimPajama数据集[51]通过进一步清理和去重从RedPajama-V1中衍生出来。同样，Dolma [52]数据集包括其他专门的领域，例如代码数据集的清理版本，包括The Stack [25]、StarcoderData [31]以及RedPajama-V1的ArXiv和StackExchange部分。Zyda [56]数据集采用了类似的方法，并进一步细化了开源数据集，包括从RedPajama-V1衍生的SlimPajama数据集。 最后，ROOTS语料库[26, 27]也属于核心开源数据集，涵盖多个领域和语言。表格1概述了这些开放数据集，并对每个数据集在透明度、多功能性和规模方面的表现进行了比较。

![第二张图片](./img/2.png)
<!-- <figure>
<img src="./img/2.png">
</figure> -->

## 3.RedPajama-V1：LLaMA训练数据的开放式复现

在我们 RedPajama 数据集的第一次迭代中，我们的主要目标是复现 LLaMA 技术报告[57]中记录的训练数据。为此，我们严格遵循原始方案的描述。本节首先记录我们重建原始 LLaMA 训练语料库的过程（第3.1节）。我们重点指出了原始数据集收集描述中的不足之处，并描述了我们如何解决这些歧义。接下来，我们报告了 RedPajama-INCITE，这是一个与橡树岭国家实验室（ORNL）合作，基于此语料库训练的一系列模型（第3.2节）。我们发现，尽管生成的3B规模模型性能良好，但在7B规模上仍然与原始LLaMA-7B模型存在差距。我们假设这部分是由于需要使用FP16精度进行训练造成的。此外，这也表明原始LLaMA训练语料库构建中的一些重要细节可能缺失。

### 3.1数据处理步骤

在这里，我们描述了我们尝试重建 LLaMa 技术报告[57]中描述的训练语料库。LLaMA 训练语料库的预训练数据来自七个数据集：英文CommonCrawl、C4、GitHub、维基百科、书籍（古腾堡计划和Books3）、ArXiv和Stack Exchange。LLaMA 技术报告中对每个数据集都进行了简短的（大约一段）描述，但数据集描述中存在一些不足之处。在本节中，我们详细介绍了复现每个数据集的过程，重点指出了LLaMA技术报告中描述的不足之处，并描述了我们解决这些歧义的选择。这些步骤共同产生了一个大约1.2万亿个符元的数据集。表2总结了这些数据集和符元计数。在附录中的表10中，我们进一步列出了在数据集构建过程中遇到的所有不确定性。

![第三张图片](./img/3.png)
<!-- <figure>
<img src="./img/3.png">
</figure> -->

* **普通抓取** LLaMA语料库包含 2017 年至 2020 年的五个CommonCrawl快照，这些快照使用CCNet管道[61]进行处理。 CCNet对分片中的每个快照进行去重，并为每个快照中的数据分配质量等级。它基于在维基百科上训练的5-gram Kneser-Ney模型分配的困惑度分布，为每个文档分配“头部”、“中间”和“尾部”分类。在这里，我们只保留“头部”和“中间”部分，并丢弃“尾部”。此外，Touvron等人。[57]使用在维基百科参考文章上训练的线性分类器来过滤掉低质量文档。 LLaMA论文没有指定数据集中使用了哪些快照，也没有给出关于分类器的详细信息。

我们选择了五个英文CommonCrawl快照2019-30, 2020-05, 2021-04, 2022-5和2023-06，代表项目开始前五年中的第一个快照。为了训练维基百科参考分类器，我们在2023年4月1日前下载了最新的可用英文维基百科快照。我们从维基百科快照中提取了3800万个URL，并爬取了30万个页面。然后，我们使用CCNet管道对维基百科参考文献进行适度的清洗步骤，并使用fastText训练一个unigram分类器。最后，我们将所有得分低于0.25的文档过滤掉，这将我们的CommonCrawl数据集缩减到与LLaMA CommonCrawl数据集大致相同的大小。

* **C4** LLaMA语料库包含C4数据集[46]，以包含CommonCrawl的不同版本。 我们使用C4的c4_en版本，该版本由Allen AI在Hugging Face Hub上提供。

* **Github** LLaMA语料库使用在Google BigQuery上可用的公共GitHub数据集，并保留在Apache、BSD和MIT许可证下分发的项目。LLaMA语料库还使用一些启发式方法过滤低质量文件，并在文件级别进行去重。对于RedPajama-V1，我们使用一组基于文件长度、字母数字字符比例和文件扩展名的过滤器来删除低质量文件。我们在附录C中提供了完整的启发式算法列表。

* **维基百科** LLaMA语料库使用2022年6月至8月期间20种语言的维基百科转储，并对数据进行处理以删除超链接、注释和其他格式样板。对于RedPajama-V1，我们使用Hugging Face Hub上提供的维基百科数据集，该数据集使用2023年3月20日的转储。 这也预处理数据以删除超链接、注释和其他样板。

* **Gutenberg and Books3** LLaMA语料库使用来自古腾堡计划的图书语料库和来自Pile的Books3。我们只使用古腾堡计划的PG19子集，并使用SimHash去除近似重复项。 我们最初也包含Books3，但由于版权问题将其删除。

* **arXiv** LLaMA语料库处理arXiv LaTeX文件，并删除第一节之前的所有内容、注释、内联展开的定义和宏以及参考文献，遵循[29]。我们从Amazon S3下载了“arXiv”请求者付费存储桶中的arXiv数据，并实现了类似的后处理，只保留LaTeX源文件并删除序言、注释、参考文献，并展开宏。

* **Stack Exchange** LLaMa语料库包含Stack Exchange的转储。 数据保留来自28个最大的网站，文本中删除HTML标签，答案按得分从高到低排序。 同样，我们从互联网档案下载Stack Exchange，只保留来自28个最大网站的帖子，并删除HTML标签。此外，我们将帖子分组为问答对，并按其得分对答案进行排序。

### 3.2RedPajama-INCITE 系列大语言模型

为了评估 RedPajama-V1 与原始 LLaMA 语料库的匹配程度，我们与 Incite 项目合作，在橡树岭国家实验室的 Summit 超级计算机上训练了一系列不同规模的大语言模型。 RedPajama-Incite 系列大语言模型包括一套 3B 和 7B 模型规模的预训练和指令调优模型。在本节中，我们首先描述 Summit 超级计算机的计算设置以及对预训练运行的影响（第 3.2.1 节）。然后，我们将描述如何评估模型，并推测这些模型与 LLaMA 系列模型之间质量差异（第 3.2.2 节）。

#### 3.2.1 Summit 训练设置

在本节中，我们将描述 Summit 超级计算机以及训练 RedPajama-Incite 系列大语言模型的工程和预训练挑战。我们的语言模型使用橡树岭国家实验室的 Summit 超级计算机进行训练，这是一个包含 4608 个 6xV100 节点的集群，运行 IBM Power9 架构。这种设置在训练现代大语言模型时带来了一些挑战。 在下文中，我们将讨论这些挑战并描述我们如何克服它们。

IBM Power9 架构 使用的指令集与大多数现代芯片组（即英特尔、ARM 或苹果的芯片）不同。现代版本的 PyTorch 以及它们依赖的大部分 Python 栈都没有预编译以支持 Power9 架构（官方支持的最新版本是 PyTorch 1.9）。 为了支持使用现代库进行预训练，我们团队的成员需要从头重新编译 PyTorch 并为 Summit 构建一个自定义训练栈。 其中一些工作在 GPT-NeoX 技术报告 [6] 中有更详细的记录。

在撰写本文时，Summit 超级计算机运行在 V100 GPU 上，这比通常用于训练大语言模型的 A100 或 H100 GPU 更旧。至关重要的是，V100s 不支持 bf16 数据类型，而这对于现代稳定的 LLM 训练方案是必要的。相反，我们不得不使用 fp16 进行训练，并使用损失缩放[37]以实现稳定的训练运行。我们还必须降低学习率，这与 LLaMA 训练中报告的学习率相比有所降低，这可能对收敛产生了影响（3B 模型为 $1.6 \cdot 10 ^ {-4} $ ，7B 模型为 $1.2 \cdot 10 ^{-4}$）。

IBM Power9 架构具有缓慢的互连，限制了我们每次运行可以使用的节点数量。 我们也无法使用整个集群，因为其他项目正在同时运行。 我们使用 512 个节点并行（3072 个 GPU）训练 7B 模型，使用 256 个节点并行（1536 个 GPU）训练 3B 模型，每个模型的全局批量大小为 4M 个符元。 在扩展实验中，我们发现如果不增加全局批量大小，我们就无法进一步增加并行度，这会损害收敛性。

6xV100 节点 为使用张量和流水线并行进行训练带来了挑战。 我们对 7B 模型使用了 12 路流水线并行，对 3B 模型使用了 6 路流水线并行，并且对这两个模型都使用了 2 路张量并行。

在解决这些挑战之后，我们能够在 Summit 上总共训练 3B 模型 8000 亿个符元，总共训练 7B 模型 1.001 万亿个符元。 我们在预热期之后线性衰减学习率，这与原始 LLaMA 论文中描述的学习率一致。

#### 3.2.2 评估

在这里，我们讨论了 RedPajama-INCITE-3B 和 7B 模型在常用基准测试上的评估。完整的成果和基准分数在附录C.2中提供。在训练 RedPajama-Base-INCITE-3B 达 8000 亿个符元后，它具有更好的少样本性能（在 HELM classic[9]中衡量，为 16 个核心场景的平均分数）和更好的零样本性能（使用 Eleuther AI 的 LM 评估工具[18]衡量），与其他类似规模的开放模型相比，包括广受好评的 GPT-Neo 和 Pythia-2.8B（分别使用 4200 亿和 3000 亿个符元在 Pile 上训练）。 在 HELM 上，它比这些模型高出 3-5 分。 在LMevaluation测试套件的一个子集任务上，它比这些开放模型的性能高出2-7个百分点。

在HELM-classic测试中，RedPajama-INCITE-7B-Base模型比 Falcon-7B 落后1.0个百分点，比Llama-7B落后4.1个百分点。 我们进一步细分了这些任务，发现它只在需要使用logprobs的任务上落后，而logprobs计算的是正确答案和错误答案概率之间的差异。 然而，该模型在直接生成答案并衡量质量的任务上取得了相当的平均HELM分数。 由于LM测试套件中的所有基准都使用logprobs，因此我们在这个基准测试中也看到了类似的较低结果。 我们假设这部分是由于使用FP16进行训练造成的，这使得我们无法使用更大的学习率。 此外，如上一节所示，训练数据集的构建存在不确定性来源，这可能导致数据集与用于训练Llama-1模型的数据集略有不同。 我们认为这两个因素导致了与Llama模型相比性能略低。

RedPajama-INCITE-7B-Instruct 是基础模型的指令调优版本，通过在来自P3 (BigScience) [49]和Natural Instructions (AI2) [39]的各种NLP任务上进行训练，从而优化了少样本性能。 Instruct版本在少样本任务上表现出色，在HELM上比同等规模的领先开放模型（包括Llama-7B、Falcon-7B（基础版和指令调优版）以及MPT-7B（基础版和指令调优版））高出2-8个百分点。 我们在补充材料中提供了详细的评估分数。

## 4. RedPajama-V2

与 RedPajama 数据集的第一个迭代版本相比，第二个迭代版本完全专注于网络数据，并且除了**透明度**和**规模**的设计原则外，我们还更加重视**通用性**。具体来说，除了提供完全透明和开源的数据集的目标外，语料库的目的是作为创建高质量子集的基础。通过开源数据集及其制品，实现了透明度的目标；通过处理Common Crawl语料库的大部分内容，实现了规模的目标；为了遵循多功能性的设计原则，我们将RedPajama V2发布为一个数据集，该数据集富含一组元数据，能够快速、低成本地迭代创建高质量、多样化和大型数据集。在本节中，我们首先介绍用于创建原始文本数据的数据处理步骤，概述每个文档可用的质量信号，并介绍数据集组成方面的统计数据。最后，我们介绍了关于如何利用质量信号来创建更好数据集的消融研究。

### 4.1数据处理步骤

RedPajama-V2 是一个数据集，它通过处理CommonCrawl基金会提供的网络文档创建。 由于网络数据本质上是嘈杂的，并且只能作为嵌入在HTML代码中的文本获得，因此需要对其进行处理以使其适合训练大语言模型（LLM）。 为此，用于RedPajama-V2的原始数据经过一系列基本的处理步骤，我们将对其进行更详细的解释。

#### 4.1.1数据获取

Common Crawl Archive是一个庞大的网络爬取数据存储库，可供公众免费使用。该语料库包含自2013年以来的爬取结果，并定期（双）每月更新。除了HTML格式的原始网络数据（warc）外，该存档还提供元数据（wat）和wet格式的纯文本数据。它已成为许多数据集的基础，包括C4 [46]、RefinedWeb [44]、Dolma [52]和FineWeb [43]等。

为了创建RedPajama-V2数据集，我们使用了2014年4月至2023年4月之间所有84个每月快照中提取的网络文本（即.wet文件），并将其通过CCNet管道[61]处理。与RPv1相比，这里我们保留所有困惑度区间，除了英语数据外，我们还保留法语、德语、意大利语和西班牙语数据。我们选择这个管道是因为它的处理较为轻量级，这符合我们尽可能保留原始数据集中尽可能多的信息的指导原则，并允许下游模型开发者过滤数据集。此处理步骤产生超过1000亿个单独的文本文档。

#### 4.1.2质量信号

像Llama [57, 58]、Mistral [22]、Falcon [2]、MPT [53]或Qwen [3]模型这样最先进的开源大型语言模型（LLM），这些模型的一个核心要素是这些模型所训练的大量高质量数据。例如，Llama 3 训练使用了 15 万亿个经过精心挑选的符元。提供必要规模的最突出数据来源是 CommonCrawl 公开提供的爬取数据。然而，这些原始文本（在本例中，我们还通过 CCNet 管道对其进行了处理）由于从 HTML 转换为纯文本过程中产生的伪影（例如，解析错误和菜单）、总体质量较低的数据来源以及网络内容分布固有的偏差，仍然不适合直接用作大语言模型 (LLM) 的训练数据。为了清理此类数据集，文献中提出了多种启发式方法，以从大型异构网络数据集中提取高质量数据集。然而，与之前过滤掉低质量内容的数据集不同，我们的方法保留了整个原始文本语料库，并将质量信号作为附加元数据纳入其中。此策略使我们能够使用全谱数据，将通常被丢弃的部分转换为信息属性，从而增强我们数据集的效用。这使得能够创建其他数据集，例如 C4，作为 RedPajama-V2 数据集的特例。 对于每个文档，我们都提供了在 C4 [46]、Gopher [45]、RefinedWeb [44]、“预训练者指南” [34] 和 DSIR [62] 中使用的质量信号。这些大致可以分为衡量 自然语言、文本的 重复性、基于文本 内容、基于机器学习 (ML) 的启发式方法和 去重 的质量信号。 下面，我们将详细解释这些类别中的每一个。 附录 D.2 提供了一个包含所有质量信号的详细描述以及直方图的完整列表。

* **自然语言** 从网站提取的文本文档通常包含与自然语言不符的内容，例如 JavaScript 代码、菜单和其他样板文本。为了衡量给定文本文档的自然程度，我们提供了一些简单的启发式度量，例如所有大写单词或字母的比例、以省略号结尾的行数比例、唯一词的比例、一行是否以终端标点符号结尾等等。

* **重复性** 网络数据中经常观察到的一种伪影是重复文本，这与无信息内容有关 [45]。 重复生成也是语言模型已知的失效模式[21]，去除过度重复的内容可能有助于缓解这种行为[45]。 对于每个文档，我们计算出现在最频繁（词）n 元中字符的比例 $n \in \{2, 3, 4\}$。 其次，我们计算对于 $n \in \{5, \cdots, 10\}$ 的值，出现在任何重复 n 元中的字符比例。 我们确保不会多次计算出现在重叠 n 元中的字符。

* **基于内容的** 网络文档可能包含有害和令人反感的内容，需要解决。 为此，我们提供了C4和RefinedWeb中使用的信号，即：（1）包含在LDNOOBW黑名单中的词序列数量4。 此外，我们还包含一个标志，指示文档的域名是否出现在UT1阻止的URL列表中5。 虽然这些质量信号侧重于NSFW内容，但我们认为其他基于内容的过滤器，例如域名或嵌入集群[55]，也是很有前景的方向。 在附录中的图8中，我们展示了通过嵌入聚类找到的主题分布。

* **机器学习启发式方法** 基于机器学习的质量信号围绕着衡量与高质量领域相似性的想法。 在这里，我们使用fastText分类器[24]，以及[62]中提出的重要性权重。 虽然机器学习过滤器已被证明可以提高数据集的质量（例如，[12, 57, 11]），但也有报道称它们会导致偏差或低估少数群体[15]。 RPv2中提供的fastText分类器信号是训练用于区分未过滤的RPv2数据和高质量领域的unigram词袋模型。 对于英语数据，我们使用维基百科、维基百科引用的网站、书籍和OpenWebText数据集。 对于非英语数据，我们只使用维基百科。 在[62]中提出的DSIR权重估计了单个样本对给定目标域在降维特征空间中的重要性，并且基于词语一元模型和二元模型。 这些权重定义为目标域与源域语言模型之间的对数似然比，其中我们使用与fasttext分类器相同的域。

* **重复数据删除** 人们发现，删除重复的训练数据可以提高模型的困惑度，减少记忆量，同时减少训练数据量和所需的计算量[28]。 重复数据删除也是最流行的数据集的核心组成部分之一[46, 52, 44]。 在RPv2中，我们包含了用于模糊重复数据删除的MinHash签名[10]（在不同的相似度级别），以及通过布隆过滤器[7]（错误率设置为 1%）发现的完全重复文档的ID。 6. 对于这个文档级别的重复数据删除，我们按顺序进行，从最新的转储（2023-14）开始，依次迭代后续的转储，直到到达最旧的转储（2014-15）。 附录中的图3显示了以这种方式标记为重复文档的数量概述。

### 4.2数据集统计

RPv2包含五种不同语言的1130亿个文档：英语、德语、法语、西班牙语和意大利语。如前所述，CCNet管道将数据集划分为三个桶“头部”、“中间”和“尾部”，分别对应于具有低、中和高维基百科困惑度的文档。 头部+中间分区中有328亿个文档，尾部分区中有800亿个文档。尾部文档通常比头部和中部区块的文档短（850个符元），∼1500个符元）。符元计数是基于使用Mistral [22] BPE分词器对1亿份文档的独立同分布样本进行估计的。表格3详细概述了每种语言和每个分区符元的数量。 我们在补充材料中提供了关于重复数据删除前后文档数量以及质量信号分布的更多统计数据。

![第四张图片](./img/4.png)
<!-- <figure>
<img src="./img/4..png">
</figure> -->

### 4.3数据集消融实验

在这里，我们提出了一系列数据集消融实验，目的是为了更好地理解第4.1.2节中介绍的质量信号如何影响使用不同启发式算法过滤的数据训练时，语言模型的性能。更具体地说，我们在这里提出以下问题：不同的质量过滤规则如何影响下游性能？ 我们力求进行广泛的评估，并衡量在不同下游基准测试上的性能以及多个领域上的语言建模目标。

#### 4.3.1设置

* **模型** 我们采用具有4.68亿参数和16亿参数以及2048个序列长度的仅解码器Llama-2架构[58]。两个模型都具有24层、16个注意力头，并且MLP扩展比率设置为4.0。4.68亿参数模型的隐藏维度为1024，而16亿参数模型的隐藏维度为2048。对于每个数据集，我们在4.68亿个符元上训练4.68亿参数模型，在3500亿个符元上训练16亿参数模型。 我们使用带有0.1权重衰减的AdamW [14] 优化器，最大学习率分别设置为 $5e^{-3}$ 和$5e^{-4}$，以及在最初1%的步骤中具有线性预热的余弦衰减调度。我们使用相对较小的规模，因为这使我们能够探索更广泛的过滤器，展示RedPajama中可用质量过滤器的广度。

* **硬件和训练栈** 由于其易于设置、使用以及高模型浮点运算利用率，我们使用OLMo框架7进行分布式训练，并使用FSDP [66]在多个GPU和节点之间进行并行化。 对于评估，我们使用lm-evaluation-harness。 我们最多使用5个具有Infiniband互连的H100节点来训练我们的模型。

* **评估指标** 我们力求对基准和领域进行广泛的覆盖。 同时，我们的运作规模相对较小，其中许多任务都过于困难，难以提供足够强的信号来区分数据集。

与FineWeb[43]数据集类似，我们寻找即使在这个小型模型规模下也能提供足够高信噪比的基准。在仔细考虑之后，我们最终选择了表4中的基准。在这里，我们通过以下方式呈现聚合分数：

* （1）计算基准的平均值；
* （2）归一化平均值；以及
* （3）每个数据配方的秩的归一化和。 
  
我们选择包含后者以避免对不同规模的分数进行平均。补充材料中提供了更详细的评分。为了基于目标领域的困惑度对数据集进行排序，我们遵循 Dolma [52] 中采用的方法，并采用 Paloma [35] 和 Pile [17] 验证集。

![第五张图片](./img/5.png)
<!-- <figure>
<img src="./img/5.png"
</figure> -->

#### 4.3.2结果
 
我们首先使用质量信号来实现文献中最广泛使用的过滤器中的一些。此外，我们还研究了 RPv2 中可用的 ML 启发式方法，这些方法基于 fastText 词袋 n-gram 分类器 [24] 和 DSIR 重要性权重 [62]。我们对 RPv2 的两个子集进行了消融研究，即 2023-14 爬取的数据和 2021-49 到 2023-14 的 9 次爬取的数据，这些数据使用 MinHash LSH 对单词 13-gram 进行去重，使用 128 个哈希函数、9 个波段和 13 行。 对于 1.6B 消融研究，我们过滤完整的 RPv2 数据集，然后大约采样 1T 个符元并使用相同的 Minhash 超参数对其进行去重。

**过滤器。**  我们力求涵盖广泛的质量过滤配置。目标并非在特定基准测试上优化性能，而是展示以不同方式过滤RPv2数据集会导致模型性能的巨大差异。因此，我们尝试了C4和Gopher规则的变体，并使用了RPv2中基于机器学习的质量信号。我们还使用了一个基于词数、平均行长、维基百科困惑度和维基百科参考文献分类器的自定义配置custom-rules。

**结果。** 从表5中，我们可以得出关于过滤RedPajama-V2数据集的一系列结论。首先，我们可以看到Gopher规则通常会提高性能。特别地，我们看到模糊去重和使用Gopher进行过滤在所有RPv2数据集上具有最高的聚合分数。此外，平均基准分数和归一化平均基准分数仅次于RefinedWeb，而排名分数高于RefinedWeb。附录中的每个基准测试表18、19和20显示，使用模糊去重和Gopher过滤的RPv2数据集始终处于中上水平（最低排名分数为19中的9），而RefinedWeb在Hellaswag、LAMBADA、Winogrande、MMLU和OpenBookQA上的表现更差。这表明，使用完整的Gopher规则和模糊去重（Minhash LSH）过滤RPv2可以创建一个在更广泛的任务中表现良好的数据集，优于所有其他数据集。其次，我们可以看到Gopher-natlang过滤器比Gopher-repetition过滤器性能更好。 第三，在基于模型的过滤的背景下，我们发现使用fasttext分类器和DSIR之间没有显著差异。 第四，仅使用行级C4过滤器似乎可以降低困惑度，但对聚合基准分数的影响可以忽略不计。 最后，我们注意到未经过滤的RPv2 2023-14数据集在Paloma数据集上的困惑度似乎最低，而其他过滤方法会导致模型具有更高的困惑度。 我们认为这（至少部分）可以归因于Paloma涵盖的广泛领域。 此外，Paloma还包含RPv1数据集，这可以解释在RPv1-CC上训练的模型获得的低困惑度得分。 表6进一步表明，使用完整的Gopher规则过滤的RPv2训练的模型优于仅使用Gopher-natlang规则过滤的RPv2训练的模型，并且其质量接近于在RefinedWeb数据集上训练的模型。 总之，这一系列消融研究表明了如何利用RPv2数据集中的质量信号来逐步过滤出更好的数据集。 结合其超过100万亿个符元的巨大规模，我们看到这个数据集为创建用于大语言模型预训练的高质量网络数据集提供了强大的资源。

![第六张图片](./img/6.png)
<!-- <figure>
<img src="./img/6.png">
<figcaption>
表格 5： 针对不同数据集过滤器和其他 SOTA 网络数据集的 468M 参数 LM 的评估。基准分数是从表格 3 中概述的基准中汇总的，使用 (1) 平均准确率，(2) 排名分数和 (3) 归一化平均分数。 最好的分数以粗体下划线显示。 字体，第二好的结果以粗体显示，第三好的结果以下划线斜体显示 .
</figcaption>
</figure> -->

## 5.结论

在本文中，我们介绍了RedPajama数据集。 拥有超过100万亿个符元，这些是用于预训练语言模型的最大、完全开源和透明的数据集，并且一直是许多最强大的开源大语言模型的核心组成部分。除了伴随数据集的文档外，我们还展示了如何将RedPajama-V2过滤到质量越来越高的子集的示例，从而在各种基准任务上产生不同质量水平的语言模型，并优于在其他大型预训练语料库上训练的模型。虽然这些模型相对较小，使我们能够探索更多种类的过滤器，但这也是一个局限性，需要进一步进行更大规模的探索。 我们没有针对常见基准进行彻底的去污染分析，也没有分析数据集中存在的个人身份信息，这构成了这项工作的另一个局限性。 通过发布原始的、未过滤的RedPajama-V2数据集，但同时提供一组质量信号，我们希望未来的工作将继续基于RedPajama，并提供过滤、整理和混合多个预训练语料库的新的创新方法。

## 致谢与资金披露

我们感谢加拿大CIFAR人工智能主席项目[I.R.]的支持。以及加拿大卓越研究主席项目[I.R.]的支持。这项研究得以实现，要感谢Summit超级计算机上的计算资源，这些资源是作为INCITE 2023项目奖“用于可迁移通才人工智能的可扩展基础模型”的一部分提供的。这些资源由橡树岭国家实验室的橡树岭领导计算设施提供，该设施由美国能源部科学办公室根据合同号提供支持。 DE-AC05-00OR22725。	我们衷心感谢美国国立卫生研究院（NIH）在编号为 U54EB020405（Mobilize）下的支持，以及美国国家科学基金会（NSF）在编号为 CCF2247015（硬件感知）、 CCF1763315（超越稀疏性）、 CCF1563078（从体积到速度）和 1937301（RTML）下的支持；以及美国陆军作战能力发展司令部陆军研究实验室（DEVCOM ARL）在编号为 W911NF-23-2-0184（长上下文）和 W911NF-21-2-0251（人机交互式团队）下的支持；以及美国海军研究办公室（ONR）在编号为 N000142312633（深度信号处理）下的支持；以及斯坦福大学人工智能研究所（HAI）在编号为 247183下的支持；以及恩智浦（NXP）、赛灵思（Xilinx）、法国原子能委员会电子与信息技术实验室（LETI-CEA）、英特尔（Intel）、IBM、微软（Microsoft）、NEC、东芝（Toshiba）、台积电（TSMC）、ARM、日立（Hitachi）、巴斯夫（BASF）、埃森哲（Accenture）、爱立信（Ericsson）、高通（Qualcomm）、ADI公司（Analog Devices）、谷歌云（Google Cloud）、Salesforce、道达尔（Total）、HAI-GCP云计算研究项目、斯坦福数据科学倡议（SDSI）以及斯坦福DAWN项目的成员：Meta、谷歌和 VMware 的支持。 美国政府有权出于政府目的复制和分发转载本，即使上面有任何版权声明。 本材料中表达的任何观点、发现、结论或建议均为作者的观点，不一定反映NIH、ONR或美国政府的观点、政策或认可，无论是明示的还是暗示的。

## 参考文献
- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.Gpt-4 technical report.arXiv preprint arXiv:2303.08774, 2023.
- [2] Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al.The falcon series of open language models.arXiv preprint arXiv:2311.16867, 2023.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al.Qwen technical report.arXiv preprint arXiv:2309.16609, 2023.
- [4] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al.Pythia: A suite for analyzing large language models across training and scaling.In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.
- [5] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi.Piqa: Reasoning about physical commonsense in natural language.In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.
- [6] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al.Gpt-neox-20b: An open-source autoregressive language model.arXiv preprint arXiv:2204.06745, 2022.
- [7] Burton H Bloom.Space/time trade-offs in hash coding with allowable errors.Communications of the ACM, 13(7):422–426, 1970.
- [8] Rishi Bommasani, Kevin Klyman, Shayne Longpre, Sayash Kapoor, Nestor Maslej, Betty Xiong, Daniel Zhang, and Percy Liang.The foundation model transparency index.arXiv preprint arXiv:2310.12941, 2023.
- [9] Rishi Bommasani, Percy Liang, and Tony Lee.Holistic evaluation of language models.Annals of the New York Academy of Sciences, 1525(1):140–146, 2023.
- [10] Andrei Z Broder.On the resemblance and containment of documents.In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171), pages 21–29. IEEE, 1997.
- [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.Language models are few-shot learners.Advances in neural information processing systems, 33:1877–1901, 2020.
- [12] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al.Palm: Scaling language modeling with pathways.Journal of Machine Learning Research, 24(240):1–113, 2023.
- [13] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.Think you have solved question answering? try arc, the ai2 reasoning challenge.ArXiv, abs/1803.05457, 2018.
- [14] P Kingma Diederik.Adam: A method for stochastic optimization.2014.
- [15] Jesse Dodge, Maarten Sap, Ana Marasović, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner.Documenting large webtext corpora: A case study on the colossal clean crawled corpus.arXiv preprint arXiv:2104.08758, 2021.
- [16] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al.Glam: Efficient scaling of language models with mixture-of-experts.In International Conference on Machine Learning, pages 5547–5569. PMLR, 2022.
- [17] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.The pile: An 800gb dataset of diverse text for language modeling.arXiv preprint arXiv:2101.00027, 2020.
- [18] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou.A framework for few-shot language model evaluation, 07 2024.
- [19] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al.Olmo: Accelerating the science of language models.arXiv preprint arXiv:2402.00838, 2024.
- [20] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.Measuring massive multitask language understanding.Proceedings of the International Conference on Learning Representations (ICLR), 2021.
- [21] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi.The curious case of neural text degeneration.arXiv preprint arXiv:1904.09751, 2019.
- [22] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al.Mistral 7b.arXiv preprint arXiv:2310.06825, 2023.
- [23] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu.Pubmedqa: A dataset for biomedical research question answering.In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577, 2019.
- [24] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov.Bag of tricks for efficient text classification.In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pages 427–431. Association for Computational Linguistics, April 2017.
- [25] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries.The stack: 3 tb of permissively licensed source code.Preprint, 2022.
- [26] Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, et al.The bigscience roots corpus: A 1.6 tb composite multilingual dataset.Advances in Neural Information Processing Systems, 35:31809–31826, 2022.
- [27] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al.Bloom: A 176b-parameter open-access multilingual language model.2023.
- [28] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini.Deduplicating training data makes language models better.arXiv preprint arXiv:2107.06499, 2021.
- [29] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al.Solving quantitative reasoning problems with language models.Advances in Neural Information Processing Systems, 35:3843–3857, 2022.
- [30] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al.Datacomp-lm: In search of the next generation of training sets for language models.arXiv preprint arXiv:2406.11794, 2024.
- [31] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al.Starcoder: may the source be with you!arXiv preprint arXiv:2305.06161, 2023.
- [32] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang.Towards general text embeddings with multi-stage contrastive learning.arXiv preprint arXiv:2308.03281, 2023.
- [33] Stephanie Lin, Jacob Hilton, and Owain Evans.Truthfulqa: Measuring how models mimic human falsehoods, 2021.
- [34] Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al.A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity.arXiv preprint arXiv:2305.13169, 2023.
- [35] Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, et al.Paloma: A benchmark for evaluating language model fit.arXiv preprint arXiv:2312.10523, 2023.
- [36] Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, et al.Openelm: An efficient language model family with open-source training and inference framework.arXiv preprint arXiv:2404.14619, 2024.
- [37] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al.Mixed precision training.arXiv preprint arXiv:1710.03740, 2017.
- [38] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.Can a suit of armor conduct electricity? a new dataset for open book question answering.In EMNLP, 2018.
- [39] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi.Cross-task generalization via natural language crowdsourcing instructions.In ACL, 2022.
[40]
Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela.Adversarial nli: A new benchmark for natural language understanding.In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2020.
[41]
Nomic.Structure unstructured data: Interact, discover insights and build with unstructured text, image and audio data., 2024.Accessed: 2024-06-12.
[42]
Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez.The LAMBADA dataset: Word prediction requiring a broad discourse context.In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics.
[43]
Guilherme Penedo, Hynek Kydlíček, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al.The fineweb datasets: Decanting the web for the finest text data at scale.arXiv preprint arXiv:2406.17557, 2024.
[44]
Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay.The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only.Advances in Neural Information Processing Systems, 36, 2024.
[45]
Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al.Scaling language models: Methods, analysis & insights from training gopher.arXiv preprint arXiv:2112.11446, 2021.
[46]
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.Exploring the limits of transfer learning with a unified text-to-text transformer.Journal of machine learning research, 21(140):1–67, 2020.
[47]
Siva Reddy, Danqi Chen, and Christopher D Manning.Coqa: A conversational question answering challenge.Transactions of the Association for Computational Linguistics, 7:249–266, 2019.
[48]
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi.Winogrande: an adversarial winograd schema challenge at scale.Commun. ACM, 64(9):99–106, aug 2021.
[49]
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush.Multitask prompted training enables zero-shot task generalization, 2021.
[50]
Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi.Socialiqa: Commonsense reasoning about social interactions.arXiv preprint arXiv:1904.09728, 2019.
[51]
Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Joel Hestness, Natalia Vassilieva, Daria Soboleva, and Eric Xing.Slimpajama-dc: Understanding data combinations for llm training.arXiv preprint arXiv:2309.10818, 2023.
[52]
Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al.Dolma: An open corpus of three trillion tokens for language model pretraining research.arXiv preprint arXiv:2402.00159, 2024.
[53]
MosaicML NLP Team.Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023.Accessed: 2023-05-05.
[54]
Snowflake AI Research Team.Snowflake arctic: The best llm for enterprise ai — efficiently intelligent, truly open, 2023.Accessed: 2024-05-27.
[55]
Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari Morcos.D4: Improving llm pretraining via document de-duplication and diversification.Advances in Neural Information Processing Systems, 36, 2024.
[56]
Yury Tokpanov, Beren Millidge, Paolo Glorioso, Jonathan Pilault, Adam Ibrahim, James Whittington, and Quentin Anthony.Zyda: A 1.3t dataset for open language modeling, 2024.
- [57] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.Llama: Open and efficient foundation language models.arXiv preprint arXiv:2302.13971, 2023.
- [58] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.Llama 2: Open foundation and fine-tuned chat models.arXiv preprint arXiv:2307.09288, 2023.
- [59] Ben Wang and Aran Komatsuzaki.GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.https://github.com/kingoflolz/mesh-transformer-jax, May 2021.
[60]
Johannes Welbl, Nelson F Liu, and Matt Gardner.Crowdsourcing multiple choice science questions.arXiv preprint arXiv:1707.06209, 2017.
[61]
Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave.Ccnet: Extracting high quality monolingual datasets from web crawl data.arXiv preprint arXiv:1911.00359, 2019.
[62]
Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy S Liang.Data selection for language models via importance resampling.Advances in Neural Information Processing Systems, 36:34201–34227, 2023.
[63]
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel.mT5: A massively multilingual pre-trained text-to-text transformer.In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online, June 2021. Association for Computational Linguistics.
[64]
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.Hellaswag: Can a machine really finish your sentence?In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.
[65]
Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, et al.Map-neo: Highly capable and transparent bilingual large language model series.arXiv preprint arXiv:2405.19327, 2024.
[66]
Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al.Pytorch fsdp: experiences on scaling fully sharded data parallel.arXiv preprint arXiv:2304.11277, 2023.

## 附录 A 预期用途

RedPajama 数据集创建的主要目是为大型语言模型提供训练数据。RedPajama 包含来自不同来源和领域的数据。RedPajama-V1 包含从网络抓取获得的数据、维基百科文章、从 arXiv 上的文章中提取的科学内容，以及各种编程语言的代码。RedPajama-V2 包含的数据完全基于网络抓取，并附带一系列旨在用于过滤原始数据集的质量信号。

## 附录 B数据集可访问性

RedPajama-V1 和 RedPajama-V2 都可以通过 Huggingface Hub 下载，链接分别为 https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T 和 https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2。

可通过公共 HTTP 端点访问。 我们也通过公共 HTTPS 端点提供对数据集的访问。 RedPajama-V1 数据集组件的网址列表可从 https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt 获取。 RedPajama-V2 不同组件的网址列表可从以下网址获取：

* 原始文本文档可从 https://data.together.xyz/redpajama-data-v2/v1.0.0/urls/document-urls.txt 获取。
* 头部和中间分区的质量信号可从 https://data.together.xyz/redpajama-data-v2/v1.0.0/urls/quality_signals-urls.txt 获取。
* 完全重复的文档 ID 列表可从 https://data.together.xyz/redpajama-data-v2/v1.0.0/urls/duplicates-urls.txt 获取。
* 最小哈希签名可从 https://data.together.xyz/redpajama-data-v2/v1.0.0/urls/minhash-urls.txt 获取。

### B.1数据集结构

RedPajama-V1 和 RedPajama-V2 都以 JSON Lines 格式分发，并被分割成多个分片。 由于两者性质不同，这两个数据集的结构也不同。

#### B.1.1 RedPajama-V1

RedPajama-V1 包含七个领域，其结构也相应地进行了划分。 除了 Common Crawl 子集之外，每个组件都遵循以下结构：

``` json
{
 "文本": "...", "meta" {... }
}
```

不同来源的元数据字段各不相同：

* Arxiv子集的元数据字段包括时间戳、年月份、arxiv_id、语言和网址。
* C4子集的元数据字段包括时间戳、来源、语言和网址。
* Github子集的元数据字段包括内容哈希值、时间戳、来源、行数、最大行长、平均行长、字母数字比例、仓库名称、ID、大小、二进制、副本、引用、路径、模式、许可证、语言。
* Stack Exchange子集的元数据字段包括时间戳、来源、语言、问题得分和网址。
* 维基百科子集的元数据字段包括时间戳、标题、语言和网址。

Common Crawl子集遵循以下结构：

``` json
{
 "文本": "...",
 "pred_label":...,
 "pred_label_prob":...,
 "wiki_prob":...,
 "来源":"..."
}
```

#### B.1.2 RedPajama-v2

该数据集的核心由文本文档、质量标注、重复ID和MinHash签名组成。对于文本文档，其结构在很大程度上遵循CCNet定义的结构。具体来说，给定CommonCrawl快照的文档被分成5000个分片，其中文件名指示分片、文档语言和困惑度区间（分区）。高质量标注、重复项和 MinHash 遵循相同的逻辑，并反映原始文档的文件名。

* **文件结构**。 包含原始文本文档的文件按照以下模式组织：

```
documents/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.json.gz
```

其中snapshot_id对应于 RPv2 中包含的任何抓取数据，shard_id范围从0000到4999，lang可以是en, de, fr, es或it。最后，ppl_bucket表示根据维基百科困惑度进行的划分，可以是head、middle或tail。类似地，质量信号、重复 ID 和 MinHash 遵循以下模式

```
quality_signals/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.signals.json.gz,
duplicates/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.duplicates.parquet,
```

和
```
minhashes/<snapshot_id>/<shard_id>/<lang>_<ppl_bucket>.minhash.parquet.
```

文档结构。 这些documents存储为 Gzip 压缩的 JSONL 文件，并遵循以下模式

```
{
"url": "..." ,
" date_download ": "..." ,
" digest ": "..." ,
" length ": ...,
" nlines ": ...,
" source_domain ": "..." ,
" title ": "..." ,
" raw_content ": "..." ,
" cc_segment ": "..." ,
" original_nlines ": ...,
" original_length ": ..,
" line_ids ": [ ...],
" language ": "..." ,
" language_score ": ...,
" perplexity ": ...,
" bucket ": "..."
}
```

质量信号结构。 这些质量信号是经过Gzip压缩的JSONL文件，并遵循以下模式

```
{
"id": "..." ,
" id_int ": ...,
" metadata ": {
" cc_segment ": "..." ,
" cc_net_source ": "..." ,
"url": "..." ,
" source_domain ": "..." ,
" language ": "..." ,
" snapshot_id ": "..."
},
"quality_signals ": {
	"key": [[ start , end, score ]]
	}
}
```

quality_signals字段是一个字典，其中质量信号的名称作为键，元组列表作为值。 每个元组包含三个浮点数start、end和score，指示raw_content字符串中score 的位置。 此表示法遵循Dolma [52] 中使用的表示法，并允许单个表示法对应用于不同文本粒度级别（例如，行级和文档级）的质量信号进行编码。

* **重复ID** 重复文档的ID存储为parquet文件。 parquet文件中的每一行对应于至少在整个语料库中重复一次的文档。 我们强调，这并不包括具有后续重复项的文档的第一次出现。 换句话说，如果删除重复列表中出现的每个文档，则每个文档集群的一个成员保留在数据集中。

* **最小哈希** 最小哈希签名存储在parquet文件中，并被划分成多个band和rows，对应于 $\{.7, 0.8, 0.9, 1.0\}$ 范围内的不同级别的Jaccard相似度。

## 附录 CRedPajama-V1
在这里，我们提供了RedPajama-V1数据集的更多细节和结果。

### C.1从GitHub获取的代码的过滤启发式方法
正如本文主要部分所述，我们通过仅保留Apache、BSD和MIT许可证下的项目来过滤原始GitHub数据集，并额外应用类似于The Stack数据集[25]中使用的过滤启发式方法。 具体来说，我们应用以下启发式方法集，删除任何具有以下属性的文件：

• 最大行长超过1000个字符。
• 平均行长超过100个字符。
• 字母数字字符的比例小于0.25。
• 字母字符数量与符元数量之比小于1.5。
• 扩展名不在以下白名单扩展名集合中： .asm, .bat, .cmd, .c, .h, .cs, .cpp, .hpp, .c++, .h++, .cc, .hh, .C, .H, .cmake, .css, .dockerfile, .f90, .f, .f03, .f08, .f77, .f95, .for, .fpp, .go, .hs, .html, .java, .js, .jl, .lua, .md, .markdown, .php, .php3, .php4, .php5, .phps, .phpt, .pl, .pm, .pod, .perl, .ps1, .psd1, .psm1, .py, .rb, .rs, .sql, .scala, .sh, .bash, .command, .zsh, .ts, .tsx, .tex, .vb, Dockerfile, Makefile, .xml, .rst, .m, .smali
C.2RedPajama-INCITE 大语言模型的详细评估
在这里，我们提供了在RedPajama-V1数据集上训练的RedPajama-INCITE 3B和7B大语言模型的详细基准分数。

Refer to caption
图2： RedPajama-INCITE-Base 3B 模型在 lm-evaluation-harness 子集上的结果。 这些任务的选择是根据评估 Pythia [4] 和 GPT-J [59] 所做的选择进行的。
表 7： RedPajama-INCITE-Base-3B-v1 模型在 lm-evaluation-harness 子集（零样本）和 HELM 上的结果，与参数数量相似的模型进行了比较。 每个基准测试中得分最高的模型以粗体显示。
Lambada
OpenAi (acc)
 	 
Hellaswag
(acc_norm)
 	 
Winogrande
(acc)
 	 
Piqa
(acc)
 	Avg.	HELM avg.
GPT-Neo	0.6223	0.5579	0.5769	0.7219	0.6197	0.3570
Pythia-2.8B	0.6466	0.5933	0.6006	0.7399	0.6451	0.3770
Pythia-2.8B-dedup	0.6524	0.5941	0.5848	0.7404	0.6429	-
RedPajama-INCITE-Base-3B-v1	0.6541	0.6317	0.6322	0.7470	0.6662	0.4060
 
表 8： RedPajama-INCITE-Base-7B-v1 模型以及指令微调后的模型在 HELM 基准测试上的结果。 每个基准测试中得分最高的模型以粗体显示。
Model	 
RedPajama 7B
Instruct
 	Llama 7B	MPT 7B	Falcon 7B	 
RedPajama 7B
Base
 	GPT J	 
Falcon 7B
Instruct
 	Pythia 7B	Dolly v2	 
MPT 7B
Instruct
 	 
Stablelm
Alpha 7B
 
HELM-AVG	0.492	0.472	0.444	0.441	0.431	0.417	0.407	0.400	0.396	0.393	0.288
MMLU - EM	0.366	0.345	0.294	0.285	0.323	0.249	0.271	0.266	0.238	0.349	0.293
BoolQ - EM	0.697	0.751	0.731	0.770	0.694	0.649	0.708	0.656	0.602	0.442	0.537
NarrativeQA - F1	0.623	0.524	0.541	0.549	0.512	0.545	0.381	0.427	0.441	0.220	0.218
NaturalQuestions (closed-book) - F1	0.229	0.297	0.284	0.289	0.258	0.156	0.192	0.141	0.133	0.247	0.077
NaturalQuestions (open-book) - F1	0.654	0.580	0.603	0.574	0.600	0.559	0.453	0.549	0.535	0.627	0.317
QuAC - F1	0.252	0.332	0.343	0.322	0.323	0.330	0.300	0.306	0.299	0.352	0.218
HellaSwag - EM	0.698	0.747	0.754	0.732	0.702	0.663	0.690	0.653	0.692	0.763	0.421
OpenbookQA - EM	0.488	0.574	0.540	0.546	0.504	0.514	0.498	0.496	0.516	0.532	0.394
TruthfulQA - EM	0.226	0.297	0.186	0.206	0.205	0.199	0.203	0.225	0.250	0.188	0.209
MS MARCO (regular) - RR@10	0.391	0.252	0.161	0.169	0.135	0.152	0.225	0.159	0.160	0.161	0.110
MS MARCO (TREC) - NDCG@10	0.709	0.482	0.369	0.362	0.322	0.345	0.481	0.342	0.359	0.387	0.253
CNN/DailyMail - ROUGE-2	0.143	0.149	0.137	0.147	0.137	0.131	0.114	0.101	0.140	0.148	0.045
XSUM - ROUGE-2	0.101	0.127	0.107	0.116	0.114	0.096	0.071	0.079	0.074	0.101	0.037
IMDB - EM	0.941	0.933	0.903	0.893	0.916	0.939	0.906	0.930	0.907	0.891	0.627
CivilComments - EM	0.667	0.578	0.525	0.511	0.536	0.520	0.516	0.527	0.520	0.270	0.490
RAFT - EM	0.682	0.583	0.618	0.586	0.611	0.619	0.498	0.542	0.466	0.616	0.368
 
表 9： RedPajama-INCITE-Base-7B-v1 模型以及指令微调模型在 LM 评估工具箱上的结果。 每个基准测试中得分最高的模型以粗体显示。
MPT 7B
Instruct
 	Falcon 7B	MPT 7B	 
RedPajama 7B
Base
 	Llama 7B	 
RedPajama 7B
Instruct
 	 
Falcon 7B
Instruct
 	Dolly v2	GPT-J	Pythia 7B	 
StableLM
Alpha 7B
 
LM-eval-harness-AVG	0.7195	0.7161	0.7100	0.6882	0.6881	0.6858	0.6813	0.6557	0.6526	0.6392	0.5260
arc_challenge (acc_norm)	0.4462	0.4326	0.4215	0.3925	0.4147	0.4078	0.4283	0.4027	0.3660	0.3532	0.2705
arc_easy (acc)	0.7218	0.7096	0.7008	0.6923	0.5253	0.7159	0.6789	0.6423	0.6225	0.6338	0.4487
boolq (acc)	0.7425	0.7361	0.7486	0.707	0.7315	0.6865	0.7089	0.6502	0.6544	0.6446	0.6006
copa (acc)	0.9000	0.8600	0.8500	0.880	0.8500	0.850	0.8400	0.8600	0.8300	0.7400	0.7500
hellaswag (acc_norm)	0.7717	0.7634	0.7626	0.7037	0.7620	0.7103	0.6978	0.6896	0.6625	0.6588	0.4122
lambada_openai (acc)	0.6918	0.7467	0.7056	0.7143	0.7360	0.6895	0.6831	0.6893	0.6831	0.6441	0.6379
piqa (acc_norm)	0.8041	0.8069	0.8052	0.7737	0.7810	0.7699	0.7856	0.7486	0.7617	0.7671	0.6736
winogrande (acc)	0.6780	0.6732	0.6859	0.6417	0.7040	0.6567	0.6669	0.6140	0.6409	0.6267	0.5012
 
C.3RedPajama-V1 数据集构建中不确定性的详细来源
表10详细概述了 RedPajama-V1 数据集构建过程中出现的各种不确定性来源。 这些不确定性主要源于[57]中关于数据集的细节缺乏。 从此列表可以看出，RedPajama-V1 和用于训练 Llama-1 模型的数据集之间可能存在不匹配。 我们认为这是导致 RedPajama-INCITE 和 LLaMA-1 性能不匹配的重要因素。

表 10： RedPajama-V1 数据集构建过程中不同不确定性和决策的概述。
Subset
 	
Uncertainty
Decision
CommonCrawl
 	
Which snapshots were used?
We use the first snapshot from 2019 to 2023.
What classifier was used, and how was it constructed?
 	
We use a fasttext classifier with unigram features and use 300k training samples.
What threshold was used to classify a sample as high quality?
 	
We set the threshold to match the token count reported in LLama.
GitHub
 	
Quality filtering heuristics
We remove any file
  •  with a maximum line length of more than 1000 characters.
  •  with an average line length of more than 100 characters.
  •  with a proportion of alphanumeric characters of less than 0.25.
  •  with a ratio between the number of alphabetical characters and the number of tokens of less than 1.5.
  •  whose extension is not in the following set of whitelisted extensions: .asm, .bat, .cmd, .c, .h, .cs, .cpp, .hpp, .c++, .h++, .cc, .hh, .C, .H, .cmake, .css, .dockerfile, .f90, .f, .f03, .f08, .f77, .f95, .for, .fpp, .go, .hs, .html, .java, .js, .jl, .lua, .md, .markdown, .php, .php3, .php4, .php5, .phps, .phpt, .pl, .pm, .pod, .perl, .ps1, .psd1, .psm1, .py, .rb, .rs, .sql, .scala, .sh, .bash, .command, .zsh, .ts, .tsx, .tex, .vb, Dockerfile, Makefile, .xml, .rst, .m, .smali
Wikipedia
 	
Which Wikipedia dump was used?
We used the most recent at the time of data curation (2023-03-20).
Books
 	
How were the books deduplicated?
We use SimHash to perform near deduplication.
 
附录 DRedPajama-V2
本节提供 RedPajama-V2 网络数据集的额外分析和统计数据，并针对在不同过滤子集上训练的消融模型呈现详细结果。

D.1去重方法的汇总统计数据
在图 3 中，我们看到 head+middle 分区中的文档数量如何随着每次爬取时间的变化而变化。 值得注意的是，直到 2018 年，文档数量相对稳定，而在 2014 年到 2016 年之间，文档数量明显减少（例如，德语文档减少了 10 倍）。 同样值得注意的是，唯一文档数量随时间的变化情况（虚线）。 具体来说，由于我们从最新的快照到最旧的快照运行去重，预期语料库中唯一文档的数量会越来越少，这可以从图 3 中观察到（注意对数刻度）。 然而，值得指出的是，在 2014 年到 2017 年之间的爬取中，唯一文档数量突然下降。 我们认为，这可以用 CommonCrawl 网络爬虫在该期间使用的种子列表不同来解释。

Refer to caption
图 3： 去重前后每个 CommonCrawl 快照的文档时间顺序计数。 去重是顺序执行的，从最新的快照开始，迭代到最旧的快照。
D.2质量信号
本节提供关于 RedPajama-V2 数据集一部分的质量信号的更多细节和统计数据。

D.2.1可用质量信号概述
质量信号集可以分为以下几类：测量自然语言（表12）、文本的重复性（表14）、基于文本内容（表15），或基于基于机器学习的启发式方法（表13）。 此外，我们还在表11中总结了由CCNet管道计算的质量信号。

表11： 来自CCNet管道的质量信号[61]。
Annotation Tag	Description
ccnet_bucket	head, middle or tail bucket of the perplexity score
ccnet_language_score	score of the language identification model
ccnet_length	number of characters
ccnet_nlines	number of lines
ccnet_original_length	number of characters before line-level deduplication
ccnet_original_nlines	number of lines before line-level deduplication
ccnet_perplexity	perplexity of an LM trained on Wikipedia
表12： 衡量文档与自然语言对应程度的质量信号摘要。
Annotation Tag	Description	Reference(s)
rps_doc_curly_bracket	 
The ratio between the number of
occurrences of ’{’ or ’}’ and the
number of characters in the raw text.
 	[46]
rps_doc_frac_all_caps_words	 
The fraction of words in the content that
only consist of uppercase letters. This is
based on the raw content.
 	[34]
rps_doc_frac_lines_end_with_ellipsis	 
The fraction of lines that end with an ellipsis,
where an ellipsis is defined as either
"…" or "U+2026".
 	[44, 45]
rps_doc_frac_no_alph_words	 
The fraction of words that contain
no alphabetical character.
 	[44, 45]
rps_doc_lorem_ipsum	 
The ratio between the number of occurrences of
’lorem ipsum’ and the number of characters in the
content after normalisation.
 	[46]
rps_doc_mean_word_length	 
The mean length of words in the content
after normalisation.
 	[44, 45]
rps_doc_stop_word_fraction	 
The ratio between the number of stop words
and the number of words in the document.
Stop words are obtained from https://github.com/6/stopwords-json.
 	[44, 45]
rps_doc_symbol_to_word_ratio	 
The ratio of symbols to words in the content. Symbols
are defined as U+0023 (#), "…", and U+2026.
 	[44, 45]
rps_doc_frac_unique_words	 
The fraction of unique words in the content.
This is also known as the degeneracy of a
text sample. Calculated based on the
normalised content.
 	[34]
rps_doc_unigram_entropy	 
The entropy of the unigram distribution of the content.
This measures the diversity of the content and is computed using
∑
x
−
x
n
⋅
log
⁡
(
1
n
)
where the sum is taken over counts of
unique words in the normalised content.
 	-
rps_doc_word_count	 
The number of words in the content after normalisation.
 	[44, 45]
rps_lines_ending_with_terminal_punctution_mark	 
Indicates whether a line ends with a terminal punctuation
mark. A terminal punctuation mark is defined as one of: ".", "!", "?", "”".
 	[46]
rps_lines_javascript_counts	 
The number of occurrences of the word "javascript" in each line.
 	[46]
rps_lines_num_words	 
The number of words in each line. This is computed based on the
normalised text.
 	[46, 44]
rps_lines_numerical_chars_fraction	 
The ratio between the number of numerical
characters and total number of characters
in each line. This is based on the
normalised content.
 	[44]
rps_lines_start_with_bulletpoint	 
Whether the lines that start with a bullet point symbol. The
following set of unicodes are considered a bullet point:
U+2022 (bullet point),
U+2023 (triangular bullet point),
U+25B6 (black right pointing triangle),
U+25C0 (black left pointing triangle),
U+25E6 (white bullet point),
U+2013 (en dash)
U+25A0 (black square),
U+25A1 (white square),
U+25AA (black small square),
U+25AB (white small square).
 	[43, 45]
rps_lines_uppercase_letter_fraction	 
The ratio between the number of uppercase letters
and total number of characters in each line.
This is based on the raw text.
 	[44]
rps_doc_num_sentences	 
The number of sentences in the content.
 	[46]
 
表13： 基于机器学习启发式的质量信号。
Annotation Tag	Description	Reference(s)
rps_doc_books_importance	 
Given a bag of 1,2-wordgram model trained on Books 
p
,
and a model trained on the source domain 
q
, This is the
logarithm of the ratio 
p
/
q
.
 	[62]
rps_doc_openwebtext_importance	 
Given a bag of 1,2-wordgram model trained on OpenWebText 
p
,
and a model trained on the source domain 
q
, this is the
logarithm of the ratio 
p
/
q
.
 	[62]
rps_doc_wikipedia_importance	 
Given a bag of 1,2-wordgram model trained on Wikipedia articles 
p
,
and a model trained on the source domain 
q
, this is the
logarithm of the ratio 
p
/
q
.
 	[62]
rps_doc_ml_wikiref_score	 
Fasttext classifier prediction for the document being a
Wikipedia reference. This is the same fasttext model
used in the RedPajama-1T dataset.
Only applies to English data.
 	[57]
rps_doc_ml_palm_score	 
Fasttext classifier prediction for the document being a
Wikipedia article, OpenWebText sample or a RedPajama-V1 book.
Only for English data.
 	[12], [16]
rps_doc_ml_wikipedia_score	 
Fasttext classifier prediction for the document being a
Wikipedia article.
This is used for non-English data
 	-
 
表14： 衡量文本重复程度的质量信号摘要。
Annotation Tag	Description	Reference(s)
rps_doc_frac_chars_dupe_10grams	The fraction of characters in duplicate word 10grams.	[43, 45]
rps_doc_frac_chars_dupe_5grams	The fraction of characters in duplicate word 5grams.	[43, 45]
rps_doc_frac_chars_dupe_6grams	The fraction of characters in duplicate word 6grams.	[43, 45]
rps_doc_frac_chars_dupe_7grams	The fraction of characters in duplicate word 7grams.	[43, 45]
rps_doc_frac_chars_dupe_8grams	The fraction of characters in duplicate word 8grams.	[43, 45]
rps_doc_frac_chars_dupe_9grams	The fraction of characters in duplicate word 9grams.	[43, 45]
rps_doc_frac_chars_top_2gram	The fraction of characters in the top word 2gram.	[43, 45]
rps_doc_frac_chars_top_3gram	The fraction of characters in the top word 3gram.	[43, 45]
rps_doc_frac_chars_top_4gram	The fraction of characters in the top word 4gram.	[43, 45]
 
表15： 基于文本内容，测量毒性的质量信号摘要。
Annotation Tag	Description	Reference(s)
rps_doc_ldnoobw_words	 
The number of sequences of words that are contained in the
List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words blocklist.
The blocklist is obtained from
https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words.
 	[46]
rps_doc_ut1_blacklist	 
A categorical id corresponding to the list of categories
of the domain of the document. Categories are obtained from https://dsi.ut-capitole.fr/blacklists/
 	[44]
 
D.2.2直方图
质量信号分布的直方图如图4、5、6和7所示。 这些统计数据来自2023年6月的快照，仅针对英文数据计算。

Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
图4： CCNet [61] 管道计算的质量信号直方图。
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
图5： 基于机器学习的质量信号直方图。
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
图6： 基于自然语言的质量信号直方图。
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
Refer to caption
图7： 用于衡量文本重复性的质量信号直方图。
D.3基于嵌入的聚类
为了基于文本文档的语义计算聚类，我们从RedPajama-V2数据集2021年4月未过滤快照中随机抽取了2,000,000个文档，并使用Alibaba-NLP gte-large-en-v1.5模型[32]计算每个文档中间8,192个符元的嵌入。 我们使用Nomic Atlas [41]进行聚类和主题建模分析。 聚类和相关主题的概述如图8所示。 6个随机抽取的文档，以及它们对应的聚类主题和每个文档中1000个字符的子串（从随机空格字符之后开始），如表16和表17所示。

Refer to caption
图8： RedPajama-V2数据集中出现的主题聚类的可视化。 这些聚类是基于200万个未过滤的2021年4月快照文档的gte-large-en-v1.5嵌入在Nomic Atlas [41]中计算的。
表 16： Nomic Atlas [41] 中的文档示例及其对应的聚类主题。
Cluster Topics
 	
Document
(broad - medium - specific)
 	
Election - Health (2) - COVID Testing
 	
immediately moving to the Purple Tier. This is the most restrictive level in the State’s effort to control the spread of COVID-19. Businesses and residents must comply with the Purple Tier restrictions by Tuesday, Nov. 17. To determine restrictions by industry, business and activity, visit: https://covid19.ca.gov/safer-economy/ Read the full news release here: www.gov.ca.gov/2020/11/16/governor-newsom-announces-new-immediate-actions-to-curb-covid-19-transmission/ Watch the Governor’s press conference during which he made the announcement today here: www.facebook.com/CAgovernor/videos/376746553637721 According to County of Orange officials, schools that have not already opened must continue with remote classes and cannot reopen in-person. Read the County’s release here: https://cms.ocgov.com/civicax/filebank/blobdload.aspx?BlobID=118441 The California Department of Public Health has also issued a travel advisory encouraging Californians to stay home or in their region and avoid non-esse
Religion/Spirituality - Gaming - Gaming (3)
 	
Top 100 Employers, and one of Canada’s Top Employers for Young People multiple years running! At Ubisoft Toronto, we look for people who are excited to create the future of games in one of the most diverse cities in the world. We believe that embracing our differences helps us build stronger creative teams and develop better games for all players. We are an equal-opportunity employer and welcome applications from all interested candidates. We strongly encourage applications from Indigenous people, racialized people, neurodivergent people, people with disabilities, people from gender and sexually diverse communities and/or people with intersectional identities. We are committed to providing reasonable accommodation for people with disability upon request. If this sounds like your kind of studio, what are you waiting for? Apply to join us now! We thank you for your interest, however, only those candidates selected for an interview will be contacted. No agencies please. Senior Game Design
Education - Golf - Rotary Meetings
 	
what’s happening. Conversely, some people rely on the newsletter. Thus, the more avenues to inform people, the better. attendance at many social functions is poor, possibly due to the limited advertising reach. In practical terms, it means that social functions may be advertised in the OOC newsletter (current practice) the schedule, as is done for outdoor activities such as hikes the OOC’s Facebook group As when social functions are advertised in the newsletter, the person organizing the social function can choose how much location information to provide, especially if it is to be held at someone’s residence. OOC bylaw Article 3, Section 9 (f) states (highlighting added) (f) Social Coordinator: Shall be responsible for coordinating all social events for Club members only, and for preparing a schedule of these outings, not to be advertised to non-members. The executive voted to amend this statement by removing the limitation per Paragraph 3 of "Article 5 - Amending Formula" of the Const
 
表 17： Nomic Atlas [41] 中的文档示例及其对应的聚类主题。
Cluster Topics
 	
Document
(broad - medium - specific)
 	
Online Privacy - Privacy Policy - Contracts
 	
shall be governed by the laws of the Federal Republic of Germany under exclusion of the UN Convention on the International Sale of Goods (CISG), without prejudice to any mandatory conflict of laws and consumer protection provisions. 11.2 If the Customer is an entrepreneur according to Sec. 14 German Civil Code (“BGB”), a legal person under public law or a special fund under public law the courts at the place of business of the vendor shall have exclusive jurisdiction in respect of all disputes arising out of or in connection with the relevant contract. 11.3 In the event that one or more provisions of the contract should be or become invalid or unenforceable, the validity of the remaining provisions shall not be affected thereby. The invalid or unenforceable provision shall be deemed to be replaced - as existent - with statutory provisions. In case of an unacceptable rigor to one of the parties, the contract shall be deemed invalid as a whole. 11.4 In case of deviations of these General
Religion/Spirituality - Film/Movie - Movie
 	
Movie of Nelson Mandela’s life premieres in South Africa Nov. 04 - Stars Idris Elba and Naomie Harris attend the premiere of "Mandela: Long Walk to Freedom," based on the autobiography of anti-apartheid icon Nelson Mandela. Matthew Stock reports.
Election - Election (2) - Healthcare (4)
 	
McAuliffe revived that language as an amendment to the budget. He also called on the General Assembly to immediately convene a special joint committee that had been created to assess the impact that repealing the ACA would have had on Virginia. The legislature will gather April 5 to consider the governor’s amendments and vetoes, but leaders said Monday that McAuliffe’s new budget language stands no better chance this time. In a joint statement, the Republican leadership of the House of Delegates said expanding Medicaid would lead to increased costs and eventually blow a hole in the state budget. “The lack of action in Washington has not changed that and in fact, the uncertainty of federal health policy underscores the need to be cautious over the long term,” the leaders, including House Speaker William J. Howell (R-Stafford) and the man selected to replace him as speaker when he retires next year, Del. Kirk Cox (R-Colonial Heights), said via email. “Virginians can barely afford our cu
 
D.4数据消融：详细评估
我们已经在本文的主要部分展示了汇总的基准分数。 在这里，我们提供更多细节，并分别报告每个任务的分数。 结果如表 18、19 和 20 所示。

表 18： 针对不同数据集过滤器和其他强大的网络数据集，对 468M 参数语言模型进行的评估。 每个指标中得分最高的数据集以 bolded underlined表示，得分第二高的数据集以粗体表示，得分第三高的数据集以 斜体下划线 表示。
Dataset	Deduplication	Rule-based	ML Heuristics	Natural Language Inference	Coref. Res.	Sentence Completion
Exact	Fuzzy	C4	Gopher	Classif.	DSIR	PPL	ANLI	ARC-c	ARC-e	Winogrande	Hellaswag	LAMBADA
C4								33.8	22.0	37.0	51.9	32.9	15.5
Dolma-v1.7 CC								33.5	24.0	38.3	49.6	32.3	17.3
FineWeb								34.0	23.4	37.7	51.8	32.8	18.1
RefinedWeb								32.8	22.6	38.3	51.9	31.6	17.8
RPv1-CC					✔ (Wiki-Ref.)			33.9	22.4	37.5	52.6	29.7	19.0
RPv2 (2023-14)								33.3	22.2	38.5	52.4	31.5	18.2
RPv2 (2023-14)	✔							33.9	22.1	38.1	50.6	31.3	18.0
RPv2 (2023-14)		✔		✔ (full)				34.1	22.3	38.3	52.2	32.1	18.7
RPv2 (2023-14)		✔	✔					33.4	22.7	38.9	51.1	32.4	17.5
RPv2 (2023-14)		✔		✔ (natlang)			Wiki-middle	33.4	24.2	37.7	49.8	33.1	19.2
RPv2 (2023-14)		✔		✔ (Rep.)			Wiki-middle	34.2	23.1	37.4	50.8	32.5	18.5
RPv2 (9 Dumps)		✔	✔					34.3	23.5	38.6	51.5	32.0	17.2
RPv2 (9 Dumps)		✔	✔	✔ (full)				33.5	23.3	38.4	50.2	32.8	16.8
RPv2 (9 Dumps)		✔	✔	✔ (Rep.)		✔ (Palm-mix)		33.8	21.9	38.0	52.5	32.0	17.3
RPv2 (9 Dumps)		✔	✔	✔ (Rep.)	✔ (Palm-mix)			34.6	23.3	38.6	52.2	32.7	16.4
RPv2 (9 Dumps)		✔	✔	✔ (natlang)	✔ (Palm-mix)			34.8	23.0	39.2	53.0	32.3	16.9
RPv2 (9 Dumps)		✔	✔ (line-filter)	✔ (natlang)	✔ (Palm-mix)			33.7	22.9	38.5	50.9	32.3	19.9
RPv2 (9 Dumps)		✔	custom-rules	✔ (Wiki-Ref.)		
P
wiki
>
30
33.2	23.0	37.9	49.6	30.1	18.7
RPv2 (9 Dumps)		✔	custom-rules + Gopher-Rep	✔ (Wiki-Ref.)		
P
wiki
>
30
33.0	23.8	38.9	50.5	30.0	18.9
 
表 19： 对 468M 参数语言模型在 MMLU 及其子任务上进行的 5-shot 设置评估。 每个指标中得分最高的数据集以 粗体下划线表示，得分第二高的数据集以粗体表示，得分第三高的数据集以 斜体下划线 表示。
Dataset	Deduplication	Rule-based	ML Heuristics	MMLU	Stem	Humanities	Other	Social Sciences
Exact	Fuzzy	C4	Gopher	Classif.	DSIR	PPL
C4								24.9	26.4	24.1	25.8	23.4
Dolma-v1.7 CC								26.0	27.8	24.5	26.2	26.1
FineWeb								26.2	25.4	25.1	25.8	29.3
RefinedWeb								24.8	23.9	23.7	26.5	25.6
RPv1-CC					✔ (Wiki-Ref.)			25.1	25.1	23.7	24.0	28.5
RPv2 (2023-14)								26.3	26.7	25.3	24.1	29.6
RPv2 (2023-14)	✔							26.4	26.8	25.3	25.2	28.8
RPv2 (2023-14)		✔		✔ (full)				27.0	28.8	24.8	25.6	30.0
RPv2 (2023-14)		✔	✔					25.4	27.8	24.1	26.1	24.1
RPv2 (2023-14)		✔		✔ (natlang)			Wiki-middle	26.1	27.4	25.2	24.6	27.7
RPv2 (2023-14)		✔		✔ (Rep.)			Wiki-middle	25.5	24.3	25.2	27.8	24.8
RPv2 (9 Dumps)		✔	✔					26.3	28.3	25.3	25.8	26.6
RPv2 (9 Dumps)		✔	✔	✔ (full)				25.6	28.0	25.1	24.9	24.4
RPv2 (9 Dumps)		✔	✔	✔ (Rep.)		✔ (Palm-mix)		24.4	26.9	23.7	24.8	22.7
RPv2 (9 Dumps)		✔	✔	✔ (Rep.)	✔ (Palm-mix)			24.9	26.1	24.0	26.3	23.8
RPv2 (9 Dumps)		✔	✔	✔ (natlang)	✔ (Palm-mix)			25.3	27.8	24.2	25.4	24.5
RPv2 (9 Dumps)		✔	✔ (line-filter)	✔ (natlang)	✔ (Palm-mix)			25.1	27.5	24.0	25.0	24.4
RPv2 (9 Dumps)		✔	custom-rules	✔ (Wiki-Ref.)		
P
wiki
>
30
27.0	27.9	25.1	26.0	30.0
RPv2 (9 Dumps)		✔	custom-rules + Gopher-Rep	✔ (Wiki-Ref.)		
P
wiki
>
30
25.9	25.8	24.3	27.1	27.2
 
表 20： 对 468M 参数语言模型在多项选择任务上的评估。 每个指标中得分最高的的数据集以粗体下划线显示，得分第二高的数据集以粗体显示，得分第三高的数据集以斜体下划线显示。
Dataset	Deduplication	Rule-based	ML Heuristics	CoQA	OpenbookQA	PIQA	PubMedQA	SciQ	SocialIQA	TruthfulQA
Exact	Fuzzy	C4	Gopher	Classif.	DSIR	PPL
C4								3.8	30.2	64.4	46.0	51.7	33.4	33.3
Dolma-v1.7 CC								5.2	28.2	65.3	42.6	55.2	31.6	33.2
FineWeb								9.0	29.4	64.5	41.4	54.3	32.4	33.5
RefinedWeb								13.2	28.6	64.4	52.2	56.4	32.8	33.3
RPv1-CC					✔ (Wiki-Ref.)			11.6	25.4	57.3	40.6	56.7	33.1	33.9
RPv2 (2023-14)								12.5	29.2	61.6	40.8	53.0	32.9	31.4
RPv2 (2023-14)	✔							11.8	27.6	61.1	43.6	53.7	32.5	33.4
RPv2 (2023-14)		✔		✔ (full)				11.3	28.8	62.8	51.0	53.9	32.6	32.6
RPv2 (2023-14)		✔	✔					5.8	28.8	63.4	49.6	54.7	36.6	33.8
RPv2 (2023-14)		✔		✔ (natlang)			Wiki-middle	11.3	28.4	63.5	49.6	53.6	32.8	33.4
RPv2 (2023-14)		✔		✔ (Rep.)			Wiki-middle	11.9	29.4	63.1	52.6	53.4	32.5	31.6
RPv2 (9 Dumps)		✔	✔					6.6	29.0	62.0	36.2	53.7	33.2	34.3
RPv2 (9 Dumps)		✔	✔	✔ (full)				5.8	28.6	62.8	51.2	54.8	34.4	31.2
RPv2 (9 Dumps)		✔	✔	✔ (Rep.)		✔ (Palm-mix)		6.0	29.4	61.6	45.4	52.2	33.4	33.1
RPv2 (9 Dumps)		✔	✔	✔ (Rep.)	✔ (Palm-mix)			5.4	29.4	62.5	45.0	51.7	34.0	33.7
RPv2 (9 Dumps)		✔	✔	✔ (natlang)	✔ (Palm-mix)			4.9	28.0	62.9	52.8	52.0	33.0	33.6
RPv2 (9 Dumps)		✔	✔ (line-filter)	✔ (natlang)	✔ (Palm-mix)			6.4	27.0	63.2	47.8	52.9	32.8	32.0
RPv2 (9 Dumps)		✔	custom-rules	✔ (Wiki-Ref.)		
P
wiki
>
30
10.0	27.8	59.6	41.2	55.8	33.3	32.0
RPv2 (9 Dumps)		✔	custom-rules + Gopher-Rep	✔ (Wiki-Ref.)		
P
wiki
>
30
9.3	28.0	59.2	43.4	54.9	33.0	33.3
 
D.516亿参数模型的评估
表格21、22和23显示了使用16亿参数模型的消融实验结果。 每个模型都使用3500亿个符元进行训练。

表格 21： 使用3500亿个符元在不同数据集上训练的16亿参数语言模型的下游任务准确率。
Dataset	 
Fuzzy
Deduplication
 	Rule-based	ML Heuristics	Natural Language Inference	Coref. Res.	Sentence Completion
C4	Gopher	ANLI	ARC-c	ARC-e	Winogrande	Hellaswag	LAMBADA
RefinedWeb					33.6	26.9	51.7	54.4	55.8	47.9
RPv2 (full)	✔		✔	WikiRef	32.4	27.9	51.3	56.4	47.4	47.4
RPv2 (full)	✔	✔	✔(natlang)	Palm-Mix	33.6	28.7	52.4	54.5	53.1	42.9
 
表格 22： 在MMLU及其子任务上，针对16亿参数语言模型的5-shot设置下的评估结果。
Dataset	 
Fuzzy
Deduplication
 	Rule-based	ML Heuristics	MMLU
C4	Gopher	MMLU	Stem	Humanities	Other	Social Sciences
RefinedWeb					25.3	24.9	24.9	27.0	24.7
RPv2 (full)	✔		✔	WikiRef	25.2	26.0	26.7	23.9	23.3
RPv2 (full)	✔	✔	✔(natlang)	Palm-Mix	24.7	25.7	25.4	23.8	23.4
 
表格 23： 针对16亿参数语言模型在多项选择任务上的评估结果。
Dataset	 
Fuzzy
Deduplication
 	Rule-based	ML Heuristics	CoQA	OpenbookQA	PIQA	PubMedQA	SciQ	SocialIQA	TruthfulQA
C4	Gopher
RefinedWeb					47.4	31.6	73.8	57.0	75.3	41.0	36.6
RPv2 (full)	✔		✔	WikiRef	43.7	32.6	67.4	55.6	72.7	40.4	36.9
RPv2 (full)	✔	✔	✔(natlang)	Palm-Mix	22.1	32.2	71.3	55.2	71.0	42.2	35.7
 
附录 E作者责任声明
此聚合数据集根据ODC-By-1.0的条款以及可能适用于其组成部分的任何许可授予您许可。

虽然我们已尽一切努力确保此数据集中包含的数据的准确性和合法性，但由于其规模，我们无法保证其绝对完整性或正确性。 因此，如果使用此数据集违反了任何权利（法律或其他权利），包括但不限于版权侵犯、隐私侵犯或敏感信息滥用，我们作者对这些违规行为概不负责。 此数据集按“现状”提供，不附带任何明示或暗示的保证。

使用此数据集即表示您同意，因使用此数据集而产生的任何后果（法律或其他后果）均由您自行承担责任。 您承认您将在使用数据集时尽职尽责，并遵守所有适用的法律、法规和道德准则。 通过访问、下载或使用此数据集，您表示接受本声明，并承诺遵守许可条款和条件。

附录 F许可
GitHub 代码库中提供的代码 8 采用 Apache 2.0 许可证分发。 对于数据集本身，我们参考 Common Crawl 基金会使用条款 9 （对于从 Common Crawl 存档派生的数据集）。 对于其他数据集，我们参考数据集最初分发的许可证。 具体来说，

• C4 数据集位于 https://huggingface.co/datasets/allenai/c4#license，
• GitHub 子集仅限于 MIT、BSD 或 Apache 许可证，
• arxiv 子集的使用条款位于 https://info.arxiv.org/help/api/tou.html，
• 任何维基百科派生数据的维基百科许可证 https://huggingface.co/datasets/legacy-datasets/wikipedia#licensing-information，
• StackExchange 数据在互联网档案馆上的 StackExchange 许可证 https://archive.org/details/stackexchange。
我们进一步要求用户遵守他们使用的子集的每个单独许可证。