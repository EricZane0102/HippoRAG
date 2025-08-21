from ...utils.llm_utils import convert_format_to_template

# 系统提示：指导LLM如何从HTML表格中提取知识三元组
table_extraction_system = """你是一个专业的知识图谱构建专家。请从给定的HTML表格中提取知识三元组。

重要规则：
1. 为表格中每个有意义的数据点创建"虚拟交叉节点"，节点名格式：{行标识}_{列路径}_交叉点
2. 交叉节点应包含完整的行列上下文信息
3. 保持表格的层次结构信息，处理rowspan/colspan
4. 生成便于问答的语义关系
5. 提取的实体必须来源于原表格内容

输出格式：
每行一个三元组，格式为：(主体, 关系, 客体)

注意：
- 虚拟交叉节点连接表格的行列信息
- 创建直接的语义关系便于检索
- 保留数值的单位和上下文
"""

# 示例1：功耗表格（标准表格）
example1_table = """<table>
  <tr><th>芯片型号</th><th>深睡模式</th><th>正常模式</th></tr>
  <tr><td>F450</td><td>10mW</td><td>200mW</td></tr>
  <tr><td>F460</td><td>12mW</td><td>220mW</td></tr>
</table>"""

example1_output = """(F450_深睡模式_交叉点, 芯片型号, F450)
(F450_深睡模式_交叉点, 工作模式, 深睡模式)
(F450_深睡模式_交叉点, 功耗值, 10mW)
(F450_深睡模式_交叉点, 完整描述, F450在深睡模式下功耗为10mW)

(F450_正常模式_交叉点, 芯片型号, F450)
(F450_正常模式_交叉点, 工作模式, 正常模式)
(F450_正常模式_交叉点, 功耗值, 200mW)
(F450_正常模式_交叉点, 完整描述, F450在正常模式下功耗为200mW)

(F460_深睡模式_交叉点, 芯片型号, F460)
(F460_深睡模式_交叉点, 工作模式, 深睡模式)
(F460_深睡模式_交叉点, 功耗值, 12mW)

(F460_正常模式_交叉点, 芯片型号, F460)
(F460_正常模式_交叉点, 工作模式, 正常模式)
(F460_正常模式_交叉点, 功耗值, 220mW)

(F450, 深睡模式功耗, 10mW)
(F450, 正常模式功耗, 200mW)
(F460, 深睡模式功耗, 12mW)
(F460, 正常模式功耗, 220mW)
(F450, 功耗数据点, F450_深睡模式_交叉点)
(F450, 功耗数据点, F450_正常模式_交叉点)
(F460, 功耗数据点, F460_深睡模式_交叉点)
(F460, 功耗数据点, F460_正常模式_交叉点)"""

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
(UART6SPEN, 寄存器类型, 时钟控制寄存器)
(UART6SPEN, 影响模块, UART6)
(UART6SPEN, 工作模式, 睡眠模式)

(UART6SPEN_值0_交叉点, 寄存器名, UART6SPEN)
(UART6SPEN_值0_交叉点, 寄存器位, 30)
(UART6SPEN_值0_交叉点, 数值, 0)
(UART6SPEN_值0_交叉点, 功能描述, 在睡眠模式下关闭UART6时钟)
(UART6SPEN_值0_交叉点, 完整描述, 第30位UART6SPEN设为0时在睡眠模式下关闭UART6时钟)

(UART6SPEN_值1_交叉点, 寄存器名, UART6SPEN)
(UART6SPEN_值1_交叉点, 寄存器位, 30)
(UART6SPEN_值1_交叉点, 数值, 1)
(UART6SPEN_值1_交叉点, 功能描述, 在睡眠模式下开启UART6时钟)

(DACSPEN, 寄存器位, 29)
(DACSPEN, 寄存器类型, 时钟控制寄存器)
(DACSPEN, 影响模块, DAC)
(DACSPEN, 工作模式, 睡眠模式)

(DACSPEN_值0_交叉点, 寄存器名, DACSPEN)
(DACSPEN_值0_交叉点, 寄存器位, 29)
(DACSPEN_值0_交叉点, 数值, 0)
(DACSPEN_值0_交叉点, 功能描述, 在睡眠模式下关闭DAC时钟)

(DACSPEN_值1_交叉点, 寄存器名, DACSPEN)
(DACSPEN_值1_交叉点, 寄存器位, 29)
(DACSPEN_值1_交叉点, 数值, 1)
(DACSPEN_值1_交叉点, 功能描述, 在睡眠模式下开启DAC时钟)

(UART6, 受控于, UART6SPEN)
(DAC, 受控于, DACSPEN)
(睡眠模式, 影响寄存器, UART6SPEN)
(睡眠模式, 影响寄存器, DACSPEN)"""

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

example3_output = """(F450_温度范围_最低_交叉点, 芯片型号, F450)
(F450_温度范围_最低_交叉点, 参数类别, 温度范围)
(F450_温度范围_最低_交叉点, 参数子类, 最低温度)
(F450_温度范围_最低_交叉点, 数值, -40°C)
(F450_温度范围_最低_交叉点, 完整描述, F450的最低工作温度为-40°C)

(F450_温度范围_最高_交叉点, 芯片型号, F450)
(F450_温度范围_最高_交叉点, 参数类别, 温度范围)
(F450_温度范围_最高_交叉点, 参数子类, 最高温度)
(F450_温度范围_最高_交叉点, 数值, 85°C)

(F450_电压范围_最低_交叉点, 芯片型号, F450)
(F450_电压范围_最低_交叉点, 参数类别, 电压范围)
(F450_电压范围_最低_交叉点, 参数子类, 最低电压)
(F450_电压范围_最低_交叉点, 数值, 1.8V)

(F450_电压范围_最高_交叉点, 芯片型号, F450)
(F450_电压范围_最高_交叉点, 参数类别, 电压范围)
(F450_电压范围_最高_交叉点, 参数子类, 最高电压)
(F450_电压范围_最高_交叉点, 数值, 3.6V)

(F450, 最低工作温度, -40°C)
(F450, 最高工作温度, 85°C)
(F450, 最低工作电压, 1.8V)
(F450, 最高工作电压, 3.6V)
(F450, 温度范围, -40°C到85°C)
(F450, 电压范围, 1.8V到3.6V)"""

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