import ast
import re
from typing import List

import numpy as np

DAPO_ARGS = {"L_max": 7168, "L_cache": 1024}
IDX_TO_GT = {0: "[[A]]", 1: "[[B]]", 2: "[[C]]", 3: "[[D]]", 4: "[[E]]"}


def extract_boxed_contents(text: str) -> List[int]:
    r"""
    Extracts all contents within \boxed{...} from a given text string,
    after normalizing braces.
    """
    # Match \boxed{...} with non-greedy content
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, text)
    try:
        matches = ast.literal_eval(matches.group(1))
        assert isinstance(matches, list) and all(isinstance(x, int) for x in matches)
    except Exception:
        matches = []
    return matches


def soft_overlong_punishment(completion_ids, **kwargs):
    # taken from https://github.com/huggingface/trl/issues/3130
    rewards = []
    for ids in completion_ids:
        completion_length = len(ids)
        if completion_length <= DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"]:
            rewards.append(0)
        elif DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"] < completion_length <= DAPO_ARGS["L_max"]:
            rewards.append((DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"] - completion_length) / DAPO_ARGS["L_cache"])
        else:
            rewards.append(-1)
    return rewards


def format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n(.*?)$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.0 if match else -1.0 for match in matches]


def list_reward(completions, chosen_answer, **kwargs):
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, chosen_answer)]


def list_reward_with_distance(completions, pass_rates, num_candidates, **kwargs):
    rewards = []
    contents = [c[0]["content"].split("</think>")[-1].strip() for c in completions]
    for completion, pass_rate, num_candidate in zip(contents, pass_rates, num_candidates):
        # Generate possible answers up to num_candidates
        possible_answers = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"][:num_candidate]
        possible_rewards = [1, -0.8, -0.6, -0.4, -0.2][:num_candidate]
        possible_rewards = sorted(possible_rewards, reverse=True)
        # Map answers to their pass rates
        answer_to_pr = {x: y for x, y in zip(possible_answers, pass_rate)}
        # Sort by pass rate (descending)
        answer_to_pr = dict(sorted(answer_to_pr.items(), key=lambda item: item[1], reverse=True))
        # Use this ordered dictionary to assign rewards - highest (chosen) gets 1 and worst gets -0.8
        answer_to_reward = {x:y for x,y in zip(answer_to_pr.keys(), possible_rewards)}
        
        rewards.append(answer_to_reward.get(completion, -1))

    return rewards


def list_score_correctness(completions, chosen_position, **kwargs):
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    # contents is something like [\boxed{[8,4,3]}, \boxed{[3,7,1,1]},...]
    contents = [extract_boxed_contents(completion) for completion in contents]
    # contents is something like [[8,4,3], [3,1,1,7]]
    model_verdicts = [x.index(max(x)) for x in contents]
    # model_verdicts is something like [0, 3]
    return [1.0 if mv == cp else 0.0 for mv, cp in zip(model_verdicts, chosen_position)]


def list_score_max10(completions, chosen_position, **kwargs):
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    # contents is something like [\boxed{[8,4,3]}, \boxed{[3,7,1,1]},...]
    contents = [extract_boxed_contents(completion) for completion in contents]
    # contents is something like [[8,4,3], [3,1,1,7]]
    model_verdicts = [x.index(max(x)) for x in contents]
    # model_verdicts is something like [0, 3]
    model_max_scores = [max(x) for x in contents]
    # model_max_scores is something like [8, 7]
    return [1.0 if mv == cp and ms == 10 else 0.0 for mv, cp, ms in zip(model_verdicts, chosen_position, model_max_scores)]


def judgelrm_content_reward(completions, pass_rates, **kwargs):
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    # contents is something like [\boxed{[8,4]}, \boxed{[3,7]},...]
    contents = [extract_boxed_contents(completion) for completion in contents]
    # contents is something like [[8,4], [3,7]]
    content_rewards = []
    for content, pass_rate in zip(contents, pass_rates):
        if not len(content) == 2:
            content_rewards.append(0)
            continue
        score_a = content[0]
        score_b = content[1]
        gt_score_a = pass_rate[0]
        gt_score_b = pass_rate[1]

        if np.sign(score_a - score_b) == np.sign(gt_score_a - gt_score_b):
            r_rel = 2.0
        else:
            r_rel = -1.5
        if abs(score_a - gt_score_a) + abs(score_b - gt_score_b) == 0:
            r_abs = 1.0
        elif r_rel == 2 and abs(score_a - gt_score_a) + abs(score_b - gt_score_b) <= 2:
            r_abs = 0.6
        else:
            r_abs = 0.0

        if r_rel == 2 and abs(score_a - score_b) + abs(gt_score_a - gt_score_b) <= 1:
            r_conf = 0.2
        else:
            r_conf = 0.0
        content_rewards.append(r_rel + r_abs + r_conf)
    return content_rewards


def judgelrm_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n(.*?)$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    verdicts = [extract_boxed_contents(x.split("</think>")[-1].strip()) for x in completion_contents]
    format_rewards = []
    for match, verdict in zip(matches, verdicts):
        if match and len(verdict) == 2 and all(0 <= x <= 10 for x in verdict):
            format_rewards.append(1.0)
        elif not match and len(verdict) == 2 and all(0 <= x <= 10 for x in verdict):
            format_rewards.append(-0.5)
        else:
            format_rewards.append(-1.0)
    return format_rewards
