"""
experiments/green_prompt.py
Created on October 11, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import re
from typing import List, Tuple

CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
_CATEGORIES = ["Clinically Significant Errors", "Clinically Insignificant Errors", "Matched Findings"]
_SUB = ["(a) ", "(b) ", "(c) ", "(d) ", "(e) ", "(f) "]


def make_prompt(text1: str, text2: str, max_len: int = 300) -> str:
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt


def clean_responses(response: str) -> str:
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if ("[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>") in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def parse_error_counts(text: str, category: str) -> Tuple[int, List[int]]:
    m = re.search(rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)", text, re.DOTALL)
    sub = [0] * 6
    if not m or m.group(1).startswith("No"):
        return 0, sub
    if category == "Matched Findings":
        counts = re.findall(r"^\b\d+\b(?=\.)", m.group(1))
        return (int(counts[0]) if counts else 0), sub
    prefixes = _SUB
    matches = sorted(re.findall(r"\([a-f]\) .*", m.group(1)))
    if not matches:
        matches = sorted(re.findall(r"\([1-6]\) .*", m.group(1)))
        prefixes = [f"({i}) " for i in range(1, 7)]
    for pos, pre in enumerate(prefixes):
        for line in matches:
            if line.startswith(pre):
                c = re.findall(r"(?<=: )\b\d+\b(?=\.)", line)
                if c:
                    sub[pos] = int(c[0])
    return sum(sub), sub


def compute_green(response: str) -> float:
    _, sig = parse_error_counts(response, _CATEGORIES[0])
    matched, _ = parse_error_counts(response, _CATEGORIES[2])
    if matched == 0:
        return 0.0
    return matched / (matched + sum(sig))
