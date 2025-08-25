from ...utils.llm_utils import convert_format_to_template

# 系统提示：指导LLM如何从HTML表格中提取知识三元组
table_extraction_system = """你是一个专业的知识图谱构建专家。请从给定的HTML表格中提取精简且有效的知识三元组。

核心原则：
1. 优先提取直接、明确的语义关系
2. 每个数据单元只创建必要的三元组，避免冗余
3. 使用描述性的关系词，便于自然语言查询
4. 保留关键的层次结构信息
5. 考虑表格的上下文和结构特征

提取策略：
- 对于简单数据：直接创建 (实体, 属性关系, 值) 三元组
- 对于复杂关系：使用语义化的交叉节点，格式：{行实体}_{关键属性}
- 表头信息：创建 (实体, 分类/归属, 表头概念) 三元组
- 数值关系：明确单位和量化关系
- 避免创建纯粹的结构性三元组（如"数据点"、"包含"等）

召回优化技巧：
- 为重要概念创建多种表达形式的三元组
- 使用同义词和相关概念增强语义覆盖
- 对技术术语建立别名关系
- 创建层次化的概念关系以支持多跳推理

输出格式：
每行一个三元组，格式为：(主体, 关系, 客体)
"""

# 示例1：功耗表格（标准表格）
example1_table = """<table>
  <tr><th>芯片型号</th><th>深睡模式</th><th>正常模式</th></tr>
  <tr><td>F450</td><td>10mW</td><td>200mW</td></tr>
  <tr><td>F460</td><td>12mW</td><td>220mW</td></tr>
</table>"""

example1_output = """(F450, 深睡模式功耗, 10mW)
(F450, 正常模式功耗, 200mW)
(F460, 深睡模式功耗, 12mW)
(F460, 正常模式功耗, 220mW)
(F450_功耗, 深睡模式, 10mW)
(F450_功耗, 正常模式, 200mW)
(F460_功耗, 深睡模式, 12mW)
(F460_功耗, 正常模式, 220mW)
(F450, 芯片类型, 低功耗芯片)
(F460, 芯片类型, 低功耗芯片)
(深睡模式, 功耗特性, 超低功耗)
(正常模式, 功耗特性, 标准功耗)"""

# 示例2：寄存器配置表（带rowspan）
example2_table = """<table>
  <tr>
    <td rowspan="2">30</td>
    <td rowspan="2">UART6SPEN</td>
    <td>0: 在睡眠模式下关闭UART6时钟</td>
  </tr>
  <tr>
    <td>1: 在睡眠模式下开启UART6时钟</td>
  </tr>
  <tr>
    <td rowspan="2">29</td>
    <td rowspan="2">DACSPEN</td>
    <td>0: 在睡眠模式下关闭DAC时钟</td>
  </tr>
  <tr>
    <td>1: 在睡眠模式下开启DAC时钟</td>
  </tr>
</table>"""

example2_output = """(UART6SPEN, 寄存器位, 30)
(UART6SPEN, 控制模块, UART6)
(UART6SPEN, 应用场景, 睡眠模式时钟控制)
(UART6SPEN_0, 功能, 在睡眠模式下关闭UART6时钟)
(UART6SPEN_1, 功能, 在睡眠模式下开启UART6时钟)

(DACSPEN, 寄存器位, 29)
(DACSPEN, 控制模块, DAC)
(DACSPEN, 应用场景, 睡眠模式时钟控制)
(DACSPEN_0, 功能, 在睡眠模式下关闭DAC时钟)
(DACSPEN_1, 功能, 在睡眠模式下开启DAC时钟)"""

# 示例3：多层表头表格
example3_table = """<table>
  <tr>
    <th rowspan="2">芯片型号</th>
    <th colspan="2">温度范围(°C)</th>
    <th colspan="2">电压范围(V)</th>
  </tr>
  <tr>
    <th>最低</th>
    <th>最高</th>
    <th>最低</th>
    <th>最高</th>
  </tr>
  <tr>
    <td>F450</td>
    <td>-40</td>
    <td>85</td>
    <td>1.8</td>
    <td>3.6</td>
  </tr>
</table>"""

example3_output = """(F450, 最低工作温度, -40°C)
(F450, 最高工作温度, 85°C)
(F450, 最低工作电压, 1.8V)
(F450, 最高工作电压, 3.6V)
(F450, 温度范围, -40°C至85°C)
(F450, 电压范围, 1.8V至3.6V)
(F450_工作条件, 温度范围, -40°C至85°C)
(F450_工作条件, 电压范围, 1.8V至3.6V)"""

# 用户输入模板
user_input_template = """现在请处理以下表格：

表格上下文：{context}

{table_html}

提取的三元组："""

# 构建完整的prompt模板
prompt_template = [
    {"role": "system", "content": table_extraction_system},
    {"role": "user", "content": f"请处理这个功耗表格：\n\n{example1_table}\n\n提取的三元组："},
    {"role": "assistant", "content": example1_output},
    {"role": "user", "content": f"请处理这个寄存器配置表格：\n\n{example2_table}\n\n提取的三元组："},
    {"role": "assistant", "content": example2_output},
    {"role": "user", "content": f"请处理这个多层表头表格：\n\n{example3_table}\n\n提取的三元组："},
    {"role": "assistant", "content": example3_output},
    {"role": "user", "content": convert_format_to_template(original_string=user_input_template, placeholder_mapping=None, static_values=None)}
]