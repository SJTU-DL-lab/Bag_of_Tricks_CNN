# 代码风格
代码风格和规范在协同开发中非常重要，在进行开发之前，请仔细阅读[Python写作指南](https://github.com/amontalenti/elements-of-python-style/tree/master/zh-cn)并尽量遵守PEP8规范。我们会利用插件对提交的代码进行检查，只有检查合格才能成功提交。以下关键规范截取自[Python写作指南](https://github.com/amontalenti/elements-of-python-style/tree/master/zh-cn)。为了避免不必要的麻烦，建议在自己的IDE中配置PEP8代码风格检查插件。

### 建议的命名规则
请务必不要使用拼音作为变量名，如果你实在不知道如何命名变量，请到[命名网站](https://unbug.github.io/codelf/)查询。
请注意，`HTTPWriter`而不是`HttpWriter`，专有名词保持大写，如API。

- Class names: `CamelCase`, and capitalize acronyms: `HTTPWriter`, not `HttpWriter`.
- 常规变量名: `lower_with_underscores`.
- 方法/函数名: `lower_with_underscores`.
- 模块文件: `lower_with_underscores.py`. (但最好是使用那些连下划线都不需要的词)
- 静态变量: `UPPER_WITH_UNDERSCORES`.
- Precompiled regular expressions: `name_re`.

其他详情参见 [the Pocoo team][pocoo].

[pocoo]: http://www.pocoo.org/internal/styleguide/

### 短docstring尽量只用一行

	# bad
	def reverse_sort(items):
	    """
	    sort items in reverse order
	    """

	# good
	def reverse_sort(items):
	    """Sort items in reverse order."""

### 用reST风格写docstring

	def get(url, qsargs=None, timeout=5.0):
	    """Send an HTTP GET request.

	    :param url: URL for the new request.
	    :type url: str
	    :param qsargs: Converted to query string arguments.
	    :type qsargs: dict
	    :param timeout: In seconds.
	    :rtype: mymodule.Response
	    """
	    return request('get', url, qsargs=qsargs, timeout=timeout)

### 可读性非常重要

- 最好不要用缩写
- 不要怕参数名太长
- 多加注释
