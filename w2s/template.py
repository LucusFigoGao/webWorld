# 两个问题打分模板
preference_2_answers_template_1 = """## SYSTEM PROMPT
You are an impartial judge helping to evaluate the helpfulness and quality of AI’s response.

## USER PROMPT
Please help me evaluate the helpfulness and quality of the responses provided by two AI assistants to the user question displayed below. You should grade a higher score for the responses that follow the user’s instructions and provide helpful information.

### Factors for Evaluation:
1. **Accurate Information**: Ensure the AI provides information that is factual and up to date.
2. **Clarity and Comprehensibility**: Check if the AI delivers information in a clear and easily understandable manner.
3. **Completeness of the Response**: Ascertain that the AI answers all aspects of the user’s query.
4. **Contextual Understanding**: The AI should demonstrate a clear understanding of the context of the user’s query.
5. **Creative Problem-Solving**: If applicable, observe if the AI proposes creative solutions to the user’s problem.
6. **Depth of Explanation**: Examine whether the AI provides detailed and in-depth responses when required.
7. **Politeness and Professionalism**: The AI should deliver responses using respectful and professional language.
8. **Reference to Reliable Sources**: If the AI claims certain facts, it should be able to refer to recognized and trusted sources.
9. **User Engagement**: The AI should engage the user effectively and pleasantly, encouraging positive user interaction.

### Scoring
A helpful and quality response should address these subjects diligently, demonstrating prowess in delivering timely, accurate, and respectful responses to users. When a response already satisfies the factors above, it has to try to bring more engaging and creative aspects. Any score should be between 1-10. If a response satisfies the factors above, its score should be higher than 5, and a less helpful response’s score should be lower than 5.

### Evaluation Procedure
Begin by offering a brief comparative analysis of the two responses. Then, present your score. As you assess, maintain objectivity, ensuring to eliminate any potential positional or length biases. Once you’ve detailed your evaluation, present your final scores in this format: **[[score1, score2]]**, where **score1** represents your assigned score for Assistant A, and **score2** stands for your assigned score for Assistant B.

### [User Question]
{}

### Assistant A's Answer
{}

### Assistant B's Answer
{}

###output

score:"""

# 评估回复质量和有用性模板 定性模板
preference_2_answers_template_2 = """You are an impartial judge helping to evaluate the helpfulness and quality of AI’s response.

Please help me evaluate the helpfulness and quality of the responses provided by two AI assistants to the user question displayed below. You should grade a higher score for the responses that follow the user’s instructions and provide helpful information.

### Factors for Evaluation:
1. **Accurate Information**: Ensure the AI provides information that is factual and up to date.
2. **Clarity and Comprehensibility**: Check if the AI delivers information in a clear and easily understandable manner.
3. **Completeness of the Response**: Ascertain that the AI answers all aspects of the user’s query.
4. **Contextual Understanding**: The AI should demonstrate a clear understanding of the context of the user’s query.
5. **Creative Problem-Solving**: If applicable, observe if the AI proposes creative solutions to the user’s problem.
6. **Depth of Explanation**: Examine whether the AI provides detailed and in-depth responses when required.
7. **Politeness and Professionalism**: The AI should deliver responses using respectful and professional language.
8. **Reference to Reliable Sources**: If the AI claims certain facts, it should be able to refer to recognized and trusted sources.
9. **User Engagement**: The AI should engage the user effectively and pleasantly, encouraging positive user interaction.

A helpful and quality response should address these subjects diligently, demonstrating prowess in delivering timely, accurate, and respectful responses to users. When a response already satisfies the factors above, it has to try to bring more engaging and creative aspects.

### Evaluation Procedure
Begin by offering a brief comparative analysis of the two responses. As you assess, maintain objectivity, ensuring to eliminate any potential positional or length biases.Thinking step by step, which answer is more helpful.Once you’ve detailed your evaluation,present your answers in the following format:###Answer A### or ###Answer B### where ###Answer A### represents Assistant A's response is more helpful, and ###Answer B### stands for Assistant B.Don't include any other comments or explanations
### [User Question]
{}

### Assistant A's Answer
{}

### Assistant B's Answer
{}

###output
**Just provide your judgment in the format specified above and don't give any other comments or explanations.**
"""

# 评估回复质量模板 定性模板
preference_2_answers_template_3 = """You are an impartial judge helping to evaluate the quality of AI’s response.

Please help me evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should grade a higher score for the responses that follow the user’s instructions.

### Factors for Evaluation:
1. **Accurate Information**: Ensure the AI provides information that is factual and up to date.
2. **Clarity and Comprehensibility**: Check if the AI delivers information in a clear and easily understandable manner.
3. **Completeness of the Response**: Ascertain that the AI answers all aspects of the user’s query.
4. **Contextual Understanding**: The AI should demonstrate a clear understanding of the context of the user’s query.
5. **Creative Problem-Solving**: If applicable, observe if the AI proposes creative solutions to the user’s problem.
6. **Depth of Explanation**: Examine whether the AI provides detailed and in-depth responses when required.
7. **Politeness and Professionalism**: The AI should deliver responses using respectful and professional language.
8. **Reference to Reliable Sources**: If the AI claims certain facts, it should be able to refer to recognized and trusted sources.
9. **User Engagement**: The AI should engage the user effectively and pleasantly, encouraging positive user interaction.

A helpful and quality response should address these subjects diligently, demonstrating prowess in delivering timely, accurate, and respectful responses to users. When a response already satisfies the factors above, it has to try to bring more engaging and creative aspects.

### Evaluation Procedure
Begin by offering a brief comparative analysis of the two responses. As you assess, maintain objectivity, ensuring to eliminate any potential positional or length biases.Thinking step by step, which answer has higher quality.Once you’ve detailed your evaluation,present your answers in the following format:###Answer A### or ###Answer B### where ###Answer A### represents Assistant A's response is more helpful, and ###Answer B### stands for Assistant B.Don't include any other comments or explanations
### [User Question]
{}

### Assistant A's Answer
{}

### Assistant B's Answer
{}

###output
**Just provide your judgment in the format specified above and don't give any other comments or explanations.**
"""

# 一个问题打分模板
preference_1_answer_template_1 = """## System
You are an impartial judge helping to evaluate the helpfulness and quality of AI’s response.

## User 
Please help me evaluate the helpfulness and quality of the response provided by AI assistant to the user question displayed below. You will grade a score for the response.

### Factors for Evaluation:
1. **Accurate Information**: Ensure the AI provides information that is factual and up to date.
2. **Clarity and Comprehensibility**: Check if the AI delivers information in a clear and easily understandable manner.
3. **Completeness of the Response**: Ascertain that the AI answers all aspects of the user’s query.
4. **Contextual Understanding**: The AI should demonstrate a clear understanding of the context of the user’s query.
5. **Creative Problem-Solving**: If applicable, observe if the AI proposes creative solutions to the user’s problem.
6. **Depth of Explanation**: Examine whether the AI provides detailed and in-depth responses when required.
7. **Politeness and Professionalism**: The AI should deliver responses using respectful and professional language.
8. **Reference to Reliable Sources**: If the AI claims certain facts, it should be able to refer to recognized and trusted sources.
9. **User Engagement**: The AI should engage the user effectively and pleasantly, encouraging positive user interaction.

### Scoring
A helpful and quality response should address these subjects diligently, demonstrating prowess in delivering timely, accurate, and respectful responses to users. When a response already satisfies the factors above, it has to try to bring more engaging and creative aspects. Any score should be between 1-10. If a response satisfies the factors above, its score should be higher than 5, and a less helpful response’s score should be lower than 5.

### Evaluation Procedure
Begin by offering a brief comparative analysis of the response. Then, present your score. As you assess, maintain objectivity, ensuring to eliminate any potential positional or length biases. Once you’ve detailed your evaluation, present your final scores in this format: **[[score]]**, where **score** represents your assigned score for Assistant.

### [User Question]
{}

### Assistant's Answer
{}

###output
**Make sure only present your final score in this format: **[[score]]**,don't include any other comments or explanations.**

score:"""

# debate_round1_template_1="""You will act as an expert debater named {player_name} engaging in a debate over answers to questions.
# You will be shown a question and two possible answers.
# You will be assigned to argue in favour of one answer while another debater named {oppsiting player} will defend the other.
# A judge, who may know less than you, will decide the winner.
# Your goal is to persuade the judge that your answer is correct.
# You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments.
# Do not exceed {word_limit} words per argument.
# A good rule of thumb is that a short paragraph and one passage take approximately 100 words.
# Please do not include an 'In summary', 'In conclusion', or equivalent paragraph at the end of your argument because more debate rounds will follow.
# The judge only sees your argument, so focus on clarity and persuasiveness in this section.
#
# Judging Criteria:
# - Clear claims are crucial, and how you integrate them into a logical and compelling argument is essential.
# - Deductions for repeating arguments.
# - Points for developing arguments further and introducing new ones.
# - Points for demonstrating misleading claims from your opponent and for convincing counterarguments.
# - Points for including good context in the opening statement.
#
# Here is the question and the two answers (the choice order is random):
# :: <question>{question}</question>
# <your_answer>{answer_defending}</your_answer>
# <opponent_answer>{answer_opposing}</opponent_answer>
#
# You have confirmed that you will argue that the answer to {question} is {answer_defending} and against {answer_opposing}.
# Here is the transcript of the debate so far:
# <transcript>{transcript}</transcript>
#
# Structure your output as follows:
# Thinking: [YOUR THINKING]
# Argument: [YOUR ARGUMENT]
#
# First, think about your argument by carrying out these steps:
# 1. Consider what extra context the judge needs to know for your answer to make sense.
# You should include this context in your argument.
# 2. Thinking step by step, plan an argument for your position.
# Finally, write your argument.
# The judge only sees the argument, so focus on clarity and persuasiveness in this section.
# Do not exceed {word_limit} words.
# Remember to output thinking and argument stages in your response."""

# 偏好辩论模板
debate_round1_template_1 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>

Structure your output as follows:
**Opinion**: ###answer_1### or ###answer_2###
**Thinking**: [YOUR THINKING]
**Argument**: [YOUR ARGUMENT]

**Firstly**, show which answer you are defending:
1.Show which answer you are defending with the format ###answer 1### or ###answer 2##,**don't give any other comments or explanations**

**Secondly**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Thinking step by step, plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# 给定观点的辩论模板
debate_round1_template_2 = """You will act as an expert debater named {} engaging in a debate over answers to questions. You will be shown a question and two possible answers. You will be assigned to argue in favour of one answer while another debater named {} will defend the other. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is correct. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow. 

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question> 
<your_answer> {} </your_answer> 
<opponent_answer> {} </opponent_answer>

You have confirmed that you will argue that the answer to {} is {} and against {}. 

Structure your output as follows:
**Thinking**: [YOUR THINKING]  
**Argument**: [YOUR ARGUMENT]

**First**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Thinking step by step, plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output thinking and argument stages in your response.

"""

# 无思维链的版本
debate_round1_template_3 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>

Structure your output as follows:
**Opinion**: [###answer_1### or ###answer_2###]
**Thinking**: [YOUR THINKING]
**Argument**: [YOUR ARGUMENT]

**Firstly**, show which answer you are defending:
1.Show which answer you are defending with the format ###answer 1### or ###answer 2##,**don't give any other comments or explanations**

**Secondly**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""
# 给定观点的辩论模板,无思维链版本
debate_round1_template_4 = """You will act as an expert debater named {} engaging in a debate over answers to questions. You will be shown a question and two possible answers. You will be assigned to argue in favour of one answer while another debater named {} will defend the other. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is correct. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow. 

**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>
You have confirmed that you will argue that the answer to question is {} and against {}. 

Structure your output as follows:
**Argument**: [YOUR ARGUMENT WITHIN 100 WORDS]

**Finally**, plan an argument for your position and write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section.Remember to output **argument** stages in your response and follow the structure.

"""

# 根据上下文给出你的辩论答案
debate_round2_template_1 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Opinion**: [###answer_1### or ###answer_2###]
**Thinking**: [YOUR THINKING]
**Argument**: [YOUR ARGUMENT WITHIN TWO SENTENCES]

**Firstly**, show which answer you are defending:
1.Show which answer you are defending with the format ###answer 1### or ###answer 2##,**don't give any other comments or explanations**

**Secondly**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Thinking step by step, plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# 列出对手言论中最大的漏洞
debate_round2_template_2 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question> 
<your_answer> {} </your_answer> 
<opponent_answer> {} </opponent_answer>

You have confirmed that you will argue that the answer to {} is {} and against {}. 

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Thinking**: [YOUR THINKING]  
**Argument**: [YOUR ARGUMENT]

**Firstly**, think about your argument by carrying out these steps:
1. List out the biggest flaws in your opponent's argument .
2. Thinking step by step , critique your opponent's argument .

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# template 1 无思维链版本
debate_round2_template_3 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Opinion**: [###answer_1### or ###answer_2###]
**Thinking**: [YOUR THINKING]
**Argument**: [YOUR ARGUMENT WITHIN TWO SENTENCES]

**Firstly**, show which answer you are defending:
1.Show which answer you are defending with the format ###answer 1### or ###answer 2##,**don't give any other comments or explanations**

**Secondly**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# 列出对手言论中最大的漏洞，无思维链版本
debate_round2_template_4 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>
You have confirmed that you will argue that the answer to question is {} and against {}. 

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Argument**: [YOUR ARGUMENT]

**Finally**, list out the biggest flaws in your opponent's argument and critique your opponent's argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **argument** stages in your response and follow the structure.

"""
#根据上下文进行辩论
debate_round3_template_1 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Opinion**: ###answer_1### or ###answer_2###
**Thinking**: [YOUR THINKING]
**Argument**: [YOUR ARGUMENT WITHIN TWO SENTENCES]

**Firstly**, show which answer you are defending:
1.Show which answer you are defending with the format ###answer 1### or ###answer 2##,**don't give any other comments or explanations**

**Secondly**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Thinking step by step, plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# 列出对手对你的论点的评价
debate_round3_template_2="""You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.


Here is the question and the two answers (the choice order is random):
<question> {} </question> 
<your_answer> {} </your_answer> 
<opponent_answer> {} </opponent_answer>

You have confirmed that you will argue that the answer to {} is {} and against {}. 

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Thinking**: [YOUR THINKING]  
**Argument**: [YOUR ARGUMENT]

**Firstly**, think about your argument by carrying out these steps:
1. List the critiques the opponent has made to your argument .
2. Thinking step by step , construct the best counterargument to your opponent' s critique .

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# template 1无思维链版本
debate_round3_template_3 = """You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Opinion**: ###answer_1### or ###answer_2###
**Thinking**: [YOUR THINKING]
**Argument**: [YOUR ARGUMENT WITHIN TWO SENTENCES]

**Firstly**, show which answer you are defending:
1.Show which answer you are defending with the format ###answer 1### or ###answer 2##,**don't give any other comments or explanations**

**Secondly**, think about your argument by carrying out these steps:
1. Consider what extra context the judge needs to know for your answer to make sense. You should include this context in your argument.
2. Plan an argument for your position.

**Finally**, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **opinion**,**thinking** and **argument** stages in your response and follow the structure.

"""

# 列出对手对你的论点的评价
debate_round3_template_4="""You will act as an expert debater engaging in a debate over answers to questions. You will be shown a question and two possible answers. Select the response better and more helpful as your argument while other debater will also choose the answer they support to defend. A judge, who may know less than you, will decide the winner. Your goal is to persuade the judge that your answer is better and helpful. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. A good rule of thumb is that a short paragraph and one passage take approximately 100 words.

Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

The judge only sees your argument, so focus on clarity and persuasiveness in this section.
**Judging Criteria:**
- Accurate Information: Ensure you provide information that is factual and up to date.
- Clarity and Comprehensibility: Check if you deliver information in a clear and easily understandable manner.
- Completeness of the Response: Ascertain that you answer all aspects of the user’s query.
- Contextual Understanding: You should demonstrate a clear understanding of the context of the user’s query.
- Creative Problem-Solving: If applicable, observe if you propose creative solutions to the user’s problem.
- Depth of Explanation: Examine whether you provide detailed and in-depth responses when required.
- Politeness and Professionalism: You should deliver responses using respectful and professional language.
- Reference to Reliable Sources: If you claim certain facts, you should be able to refer to recognized and trusted sources.
- User Engagement: You should engage the user effectively and pleasantly, encouraging positive interaction.

Here is the question and the two answers (the choice order is random):
<question> {} </question>
<answer_1> {} </answer_1>
<answer_2> {} </answer_2>
You have confirmed that you will argue that the answer to question is {} and against {}. 

Here is the transcript of the debate so far:
<transcript> 
{} 
</transcript>

Structure your output as follows:
**Argument**: [YOUR ARGUMENT]

**Finally**, list the critiques the opponent has made to your argument and construct the best counterargument to your opponent' s critique. The judge only sees the argument, so focus on clarity and persuasiveness in this section. Remember to output **argument** stages in your response and follow the structure.

"""
# 普通版本
judge_template_1 = """You are an expert judge in a debate where the participants are arguing over the correct answer to a question . Your task is to determine which debater is correct . Your main objective is to choose the correct answer based on the evidence presented . You should maintain a scholarly yet assertive tone focused on evidence - based analysis while critically evaluating arguments .

Here is the question and the two answers ( the choice order is random ) :
Question : {}
Choices :
1: {}
2: {}

Here is the transcript of the debate :
{}
End of transcript .

Only state your answer . Format it as " Answer : <1|2 >".
"""

#思维链版本
judge_template_2 = """You are an expert judge in a debate where the participants are arguing over the correct answer to a question . Your task is to determine which debater is correct . Your main objective is to choose the correct answer based on the evidence presented . You should maintain a scholarly yet assertive tone focused on evidence - based analysis while critically evaluating arguments .

Here is the question and the two answers ( the choice order is random ) :
Question : {}
Choices :
1: {}
2: {}

Here is the transcript of the debate :
{}
End of transcript .

Choose the answer you think is most likely to be correct . First explain your reasoning , considering the strengths and weaknesses of the debater' s arguments . Second, give your answer formatted as " Answer : <1|2 >".
"""
two_response_template = preference_2_answers_template_1
one_response_template = preference_1_answer_template_1

# TEMPLATE = {"1": one_response_template, "2": two_response_template}

PREFERENCE_QA_TEMPLATE = preference_2_answers_template_2

PREFERENCE_DEBATE_TEMPLATE_CHAIN = {'1': [debate_round1_template_1], '2': [debate_round2_template_1,debate_round2_template_2],
                              '3': [debate_round3_template_1,debate_round3_template_2],"4": [judge_template_1,judge_template_2]}
PREFERENCE_DEBATE_TEMPLATE={
    "1": [debate_round1_template_1,debate_round1_template_2,debate_round1_template_3,debate_round1_template_4],
    "2": [debate_round2_template_1,debate_round2_template_2,debate_round2_template_3,debate_round2_template_4],
    "3": [debate_round3_template_1,debate_round3_template_2,debate_round3_template_3,debate_round3_template_4],
    "4": [judge_template_1]
}

TEMPLATE = {
    "QA": PREFERENCE_QA_TEMPLATE,
    "DEBATE": PREFERENCE_DEBATE_TEMPLATE
}
