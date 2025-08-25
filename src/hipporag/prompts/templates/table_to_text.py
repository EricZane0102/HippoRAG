from ...utils.llm_utils import convert_format_to_template

# 基础版本的表格转文本描述系统提示
table_to_text_system_basic = """你是一个专业的技术文档分析专家。请将给定的HTML表格转换为清晰、完整的文本描述。

目标：
1. 准确描述表格的主要用途和功能
2. 详细解释表格中的数据含义和关系
3. 保留所有重要的数值和技术信息
4. 使用自然流畅的语言，便于理解和检索

描述要求：
- 首先概述表格的整体用途
- 逐一说明各列的含义和数据类型
- 描述行与行之间的关系和模式
- 突出重要的数值、范围和技术参数
- 避免冗余，但确保信息完整性

输出格式：使用自然语言段落，确保描述准确且便于后续文本检索。
"""

# 详细版本的表格转文本描述系统提示  
table_to_text_system_detailed = """你是一个专业的技术文档分析专家。请将给定的HTML表格转换为全面、结构化的文本描述。

目标：
1. 完整描述表格的业务背景和技术用途
2. 深入分析表格中数据的语义关系和技术含义
3. 提供详细的数值分析和参数解释
4. 建立数据间的关联关系，支持复杂查询
5. 使用多样化的表达方式，增强检索覆盖度

详细描述策略：
- 表格概述：用途、适用场景、技术背景
- 结构分析：表头含义、数据组织方式、层次关系
- 数据解读：重要数值、参数范围、单位说明、特殊值
- 关系分析：行间关系、列间关联、数据模式和趋势
- 技术术语：提供同义词和相关概念，增强语义覆盖
- 应用场景：说明数据的实际应用和参考价值

输出格式：
分段描述，每段聚焦一个方面，使用丰富的技术词汇和多种表达方式。
"""

# 示例1：基础描述示例
example1_table = """<table>
  <tr><th>芯片型号</th><th>深睡模式功耗</th><th>正常模式功耗</th></tr>
  <tr><td>F450</td><td>10mW</td><td>200mW</td></tr>
  <tr><td>F460</td><td>12mW</td><td>220mW</td></tr>
</table>"""

example1_basic_output = """这是一个芯片功耗对比表格，展示了不同芯片型号在两种工作模式下的功耗数据。表格包含三列：芯片型号、深睡模式功耗和正常模式功耗。

表格记录了两款芯片的功耗特性。F450芯片在深睡模式下功耗为10毫瓦，正常工作模式下功耗为200毫瓦。F460芯片在深睡模式下功耗为12毫瓦，正常工作模式下功耗为220毫瓦。

从数据可以看出，两款芯片都具有低功耗特性，深睡模式下的功耗都控制在12毫瓦以内，正常模式功耗在200-220毫瓦范围内。F450相比F460在两种模式下功耗都略低，显示出更好的功耗控制能力。"""

example1_detailed_output = """这是一个微控制器或处理器芯片的功耗性能对比技术规格表，专门用于评估不同芯片型号在低功耗应用场景中的能耗表现。该表格对于物联网设备、嵌入式系统和电池供电设备的芯片选型具有重要参考价值。

表格结构包含三个核心技术参数列。第一列芯片型号标识了具体的产品型号，第二列深睡模式功耗反映了芯片在休眠或待机状态下的最低功耗水平，第三列正常模式功耗表示芯片在标准工作状态下的功耗消耗。

详细的功耗数据分析显示：F450芯片展现出优异的低功耗特性，深睡眠模式下仅消耗10毫瓦电力，活跃工作时功耗为200毫瓦。F460芯片的功耗水平略高，睡眠功耗12毫瓦，运行功耗220毫瓦。两款芯片的功耗比值（正常模式/深睡模式）分别为20:1和18.3:1，显示出良好的功耗管理和节能设计。

从技术应用角度，这些功耗参数对于电池寿命估算、热设计和系统功耗预算具有关键意义。深睡模式的超低功耗特性使这些芯片特别适合长期运行的传感器节点、智能终端和便携式设备应用。"""

# 用户输入模板
user_input_template = """请分析以下表格并提供详细的文本描述：

表格上下文：{context}

{table_html}

请提供完整的文本描述："""

# 构建完整的prompt模板
prompt_template_basic = [
    {"role": "system", "content": table_to_text_system_basic},
    {"role": "user", "content": f"请分析这个芯片功耗表格：\n\n{example1_table}\n\n请提供完整的文本描述："},
    {"role": "assistant", "content": example1_basic_output},
    {"role": "user", "content": convert_format_to_template(original_string=user_input_template, placeholder_mapping=None, static_values=None)}
]

prompt_template_detailed = [
    {"role": "system", "content": table_to_text_system_detailed},
    {"role": "user", "content": f"请分析这个芯片功耗表格：\n\n{example1_table}\n\n请提供完整的文本描述："},
    {"role": "assistant", "content": example1_detailed_output},
    {"role": "user", "content": convert_format_to_template(original_string=user_input_template, placeholder_mapping=None, static_values=None)}
]

# 默认使用详细版本的模板
prompt_template = prompt_template_detailed