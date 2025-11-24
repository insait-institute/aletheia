LIST_REWARD_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Enclose your final answer in the format \\boxed{{X}}, where X is your chosen option among the candidates. Do not provide any explanations or additional text. Your response should be exactly one of the options enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""

LIST_REWARD_PROMPT_COT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Think carefully step-by-step before responding with one of the options in the format \\boxed{{X}}, where X is your chosen option among the candidates. Do not provide any explanations or additional text. Your response should be exactly in the specified format, without any extra characters or spaces. Anything else will be considered invalid.
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

GENRM_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Respond with one of the options in the format \\boxed{{X}}, where X is your chosen option among the candidates. Do not provide any explanations or additional text. Your response should be exactly in the specified format, without any extra characters or spaces. Anything else will be considered invalid.
"""

CODE_COMMENTING_SYSPROMPT = """
You will be given a coding question and a candidate solution. Your task is to make the code more understandable by adding documentation to the given code. Here are some guidelines to follow while adding documentation:
1. Add a docstring at the beginning of each function. It should be in the PEP-287 style for Python, Javadoc style for Java, and Doxygen style for C++.
2. Add short inline comments to each line explaining what it does
3. DO NOT modify any existing identifiers in the given code.
4. DO NOT change the underlying logic of the original code. It should execute exactly the same after adding comments.
Return the modified code as a single markdown-formatted code block. You may think step-by-step before responding with a markdown-formatted code block.
"""

CODE_TESTING_SYSPROMPT = """
You will be given a coding question and a candidate solution. You must add assertion statements to the solution to make it more robust. Here are some guidelines to follow while adding assertion statements:
1. Add assertions to check the types of function arguments.
2. Add assertions to check the input constraints mentioned in the problem statement.
3. Add intermediate checks within the code as sanity checks wherever applicable.
4. DO NOT modify any existing identifiers in the given code.
5. DO NOT change the underlying logic of the original code. It should execute exactly the same after adding assertions.
Return the modified code as a single markdown-formatted code block. You may think step-by-step before responding with a markdown-formatted code block.
"""


DEADCODE_SYSPROMPT = """
You will be given a coding question and a candidate solution. You must add garbage or unreachable code to the solution (e.g. infinite loops, unused variables, dead code). Here are some guidelines to follow while adding garbage code:
1. Enclose the added code within "# GARBAGE CODE START" and "# GARBAGE CODE END" comments for Python, and "// GARBAGE CODE START" and "// GARBAGE CODE END" comments for C++ and Java. Do not add any other comments.
2. DO NOT add trivial statements (eg: if False, while False: pass). You must customize the added code to the context of the existing code.
3. DO NOT use obvious identifier names like garbage_var, unused_function, etc. Use contextually relevant names.
4. The added code should never execute, for example by placing it within a conditional block that is always false or in a function that is never called.
5. The added code should be syntactically correct and should not introduce any compilation or runtime errors.
6. DO NOT modify any existing identifiers in the given code.
7. DO NOT change the underlying logic of the original code. It should execute exactly the same after adding garbage code.
Return the modified code as a single markdown-formatted code block. You may think step-by-step before responding with a markdown-formatted code block.
"""
