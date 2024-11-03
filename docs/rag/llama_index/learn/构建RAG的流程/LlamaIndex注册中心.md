## LlamaIndex 注册中心

我们的数据连接器通过 [LlamaHub 🦙](https://llamahub.ai/)提供。LlamaHub 包含开源数据连接器注册表，您可以轻松地将其插入任何 LlamaIndex 应用程序（以及代理工具和 Llama 包）。

## 使用模式

开始使用：

``` python
from llama_index.core import download_loader

from llama_index.readers.google import GoogleDocsReader

loader = GoogleDocsReader()
documents = loader.load_data(document_ids=[...])
```

## 内置加载器：SimpleDirectoryReader 

`SimpleDirectoryReader` 可以支持解析多种文件类型，包括 `.md`,`.pdf`, `.jpq`, `.png`, `.docx` 以及音频和视频类型。它可直接作为 LlamaIndex 的一部分使用

``` python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

## 可用连接器

直接浏览 [LlamaHub](https://llamahub.ai/)即可查看数百种可用的连接器，其中包括：

* Notion( NotionPageReader)
* Google Docs( GoogleDocsReader)
* Slack( SlackReader)
* Discord( DiscordReader)
* Apify Actors（ApifyActor）。可以抓取网络、抓取网页、提取文本内容、下载文件，包括 `.pdf`, `.jpg`, `.png`, `.docx` 等。