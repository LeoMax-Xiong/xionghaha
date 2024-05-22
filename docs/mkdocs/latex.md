# mkdocs中使用LaTex

>关键词：mkdocs, LaTex, 数学公式
>
>arithmatex

MkDocs 使用 Python Markdown 转化 Markdown 文件为 HTML 文档。Python Markdown 支持大量格式化页面的扩展。

如果需要在页面上展示LaTex数学公式，只需要在mkdocs.yaml配置中，配置插件和javascript文件即可轻松实现。

# 插件配置
* markdown_extensions
``` yaml
markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
```

* extra_javascript
``` yaml
extra_javascript:
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js

```
