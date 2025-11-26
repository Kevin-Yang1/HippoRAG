"""
此模板用于 MuSiQue 数据集的 IRCoT (Interleaved Retrieval Chain-of-Thought) 任务。
它包含一个单样本演示 (one-shot demo)，展示了如何通过多跳推理在多个文档中寻找答案。
"""

one_shot_ircot_demo_docs = (
    # 单样本演示的文档集合。
    # 包含多个维基百科段落，作为模型推理的上下文基础。
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million\n"""
)


one_shot_ircot_demo = (
    # 完整的单样本演示字符串。
    # 组合了上面的文档集合、一个示例问题 (Question) 和对应的推理过程 (Thought)。
    f"{one_shot_ircot_demo_docs}"
    "\n\nQuestion: "
    f"When was Neville A. Stanton's employer founded?"
    "\nThought: "
    f"The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
    "\n\n"
)

ircot_system = (
    # 系统提示词 (System Prompt)。
    # 定义了助手的角色和任务目标（多跳推理）。
    # 强调了逐步生成 Thought 的要求。
    # 附带了单样本演示。
    'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    "\n\n"
    f"{one_shot_ircot_demo}"
)


prompt_template = [
    # 最终的提示词模板结构。
    # 包含系统消息和用户消息占位符。
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"},
]

"""
ircot_system 内容翻译：
你是一名智能助手，擅长帮助用户在多篇文档中进行复杂的、多跳的推理。这个任务通过多个演示示例来说明，每个示例都包含一组文档、与这些文档相关的问题，以及对应的问题多跳推理的思考过程（Thought）。你的任务是在当前步骤只生成一条 Thought，不要一次性生成全部的思考步骤！如果你认为已经到了最后一步，请以 “So the answer is:” 开头给出最终答案。

Wikipedia 标题：The Last Horse
《The Last Horse》（西班牙语：El último caballo）是一部1950年的西班牙喜剧电影，由 Edgar Neville 执导，主演为 Fernando Fernán Gómez。

Wikipedia 标题：Southampton
南安普顿大学创立于 1862 年，并于 1952 年获得皇家特许成为大学，拥有超过 22,000 名学生。 在 2010 年的世界大学学术排名（ARWU）中，该校位列前 100 名研究型大学。 2010 年的 THES - QS 世界大学排名中，南安普顿大学位列世界前 80。该大学认为自己是英国前五名研究型大学之一。该校在工程科学、海洋学、化学、癌症科学、声学与振动研究、计算机科学与电子学、光电子学以及纺织保护（其纺织保护中心将在 2009 年 10 月关闭）等领域拥有全球研究声誉。这里也是南安普顿国家海洋中心（NOCS）的所在地，该中心是自然环境研究委员会资助的海洋研究核心机构。

Wikipedia 标题：Stanton Township, Champaign County, Illinois
斯坦顿镇（Stanton Township）是美国伊利诺伊州尚佩恩县的一个镇。根据 2010 年人口普查，该镇人口为 505 人，拥有 202 套住房。

Wikipedia 标题：Neville A. Stanton
Neville A. Stanton 是一位英国人因工程学和人体工程学领域贡献而知名的教授，任职于南安普顿大学。他是注册工程师（C.Eng）、注册心理学家（C.Psychol）以及注册人体工程师（C.ErgHF）。他撰写和编辑了四十多本书籍，以及三百多篇关于相关应用的同行评议期刊论文。Stanton 是英国心理学会会士、人机工程与人体因素学会会士、英国工程技术学会会员，他的研究曾发表在《Nature》等学术期刊。他还协助组织设计人机接口，例如捷豹汽车的自适应巡航控制系统。

Wikipedia 标题：Finding Nemo
《海底总动员》（Finding Nemo）院线上映宣传海报  
导演：Andrew Stanton  
制片人：Graham Walters  
编剧：Andrew Stanton、Bob Peterson、David Reynolds  
故事作者：Andrew Stanton  
主演：Albert Brooks、Ellen DeGeneres、Alexander Gould、Willem Dafoe  
音乐：Thomas Newman  
摄影：Sharon Calahan、Jeremy Lasky  
剪辑：David Ian Salter  
制作公司：华特迪士尼影业、皮克斯动画工作室  
发行公司：博伟影业  
上映日期：2003 年 5 月 30 日  
片长：100 分钟  
国家：美国  
语言：英语  
预算：9400 万美元  
票房：9.403 亿美元


Question（问题）：Neville A. Stanton 的雇主（南安普顿大学）创立于哪一年？
Thought（推理）：Neville A. Stanton 的雇主是南安普顿大学。南安普顿大学创立于 1862 年。所以答案是：1862。
"""
