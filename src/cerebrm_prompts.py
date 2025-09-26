LIST_REWARD_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Enclose your final answer in the format \\boxed{{[[X]]}}, where [[X]] is your chosen option among the candidates. Do not provide any explanations or additional text. Your response should be exactly one of the options enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""

JUDGELRM_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and two candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to assign a score between 0 and 10 to each candidate, with 10 indicating a perfect solution that passes all test cases, 5 indicating a solution that would pass some test cases but not all, and 0 indicating a solution that fails all test cases. Output your final answer in the format \\boxed{{[<score_candidate_A>, <score_candidate_B>]}}. Do not provide any explanations or additional text. Your response should be a list of exactly two numbers between 0 and 10, enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""

DS_GRM_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to assign a score between 0 and 10 to EACH candidate, with 10 indicating a perfect solution that passes all test cases, 5 indicating a solution that would pass some test cases but not all, and 0 indicating a solution that fails all test cases. Output your final answer in the format \\boxed{{[<score_candidate_A>, <score_candidate_B>, <score_candidate_C>, ...]}} depending on the number of candidates. Do not provide any explanations or additional text. Your response should be a list of numbers between 0 and 10, enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""

RAFT_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Think carefully, step-by-step before responding with one of the options in the format \\boxed{{[[X]]}}, where [[X]] is your chosen option among the candidates. Your output should look like:
Analysis: <your step by step analysis here>
Final Answer: \\boxed{{[[X]]}}
Do not provide any explanations or additional text beyond the analysis and final answer. Your response should be exactly in the specified format, without any extra characters or spaces. Anything else will be considered invalid.
"""
