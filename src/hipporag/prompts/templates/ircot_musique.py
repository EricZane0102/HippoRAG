one_shot_ircot_demo_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million\n"""
)



one_shot_ircot_demo = (
    f'{one_shot_ircot_demo_docs}'
    '\n\n问题: '
    f"When was Neville A. Stanton's employer founded?"
    '\n思考过程: '
    f"Neville A. Stanton的雇主是南安普顿大学。南安普顿大学成立于1862年。所以答案是：1862。"
    '\n\n'
)

ircot_system = (
    '你是一个智能助手，擅长帮助用户通过多个文档进行复杂的多跳推理。这个任务通过演示来说明，每个演示包含一组文档以及相关问题和多跳推理思考过程。你的任务是为当前步骤生成一个思考过程，不要一次性生成所有的思考过程！如果你认为已经到达最终步骤，请以"所以答案是："开始。'
    '\n\n'
    f'{one_shot_ircot_demo}'
)


prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"}
]
