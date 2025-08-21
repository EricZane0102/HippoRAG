ner_system = """你是一个非常有效的实体提取系统。
"""

query_prompt_one_shot_input = """请提取对解决以下问题重要的所有命名实体。
请以JSON格式放置命名实体。

问题: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""
# query_prompt_template = """
# Question: {}

# """
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "问题: ${query}"}
]