# 结合 guidance 与 Pydantic 编程
通过 LlamaIndex 结合 guidance 生成结构化数据。

使用 guidance 库，我们可以通过强制 LLM 输出 **所需** 的 token 来保证输出结构正确。
这在您使用参数量较低的模型（例如当前的开源模型）时尤其有用，否则这些模型将很难生成符合所需输出架构的有效输出。

如果您在 colab 上打开此带阿米，则可能需要安装 LlamaIndex 🦙。

``` shell
pip install llama-index-program-guidance
pip install llama-index
```

使用 `python` 的代码需要导入相关的库：

``` python
from pydantic import BaseModel
from typing import List
from guidance.llms import OpenAI

from llama_index.program.guidance import GuidancePydanticProgram
```

定义输出字段内容

``` python
class Song(BaseModel):
    title: str
    length_seconds: int


class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]
```

定义指导 pydantic 程序

program  =  GuidancePydanticProgram （
    output_cls = Album ，
    prompt_template_str = （
        “生成一个示例专辑，其中包含艺术家和歌曲列表。使用” 
        “电影{{movie_name}}作为灵感” 
    ），
    guide_llm = OpenAI （“text-davinci-003” ），
    verbose = True ，
）
运行程序以获取结构化输出。
蓝色突出显示的文本是我们指定的变量，绿色突出显示的文本由 LLM 生成。

输出 = 程序（电影名称= “闪灵” ）
生成一张示例专辑，其中包含艺术家和歌曲列表。以电影《闪灵》为灵感
```json
{
  "name": "闪灵",
  “艺术家”：“杰克·托伦斯”，
  “歌曲”：[{
  "title": "只工作不玩耍",
  "length_seconds": " 180 ",
} ， {
  "title": "瞭望酒店",
  "length_seconds": " 240 ",
} ， {
  "title": "闪灵",
  "length_seconds": " 210 ",
} ],
}
```
输出是一个有效的 Pydantic 对象，然后我们可以使用它来调用函数/API。

输出
专辑（名称='The Shining'，艺术家='Jack Torrance'，歌曲=[歌曲（标题='All Work and No Play'，长度秒数=180），歌曲（标题='The Overlook Hotel'，长度秒数=240），歌曲（标题='The Shining'，长度秒数=210）]）