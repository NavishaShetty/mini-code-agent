"""HumanEval style problems for evaluating code generation.

This contains a subset of problems similar to OpenAI's HumanEval benchmark.
Each problem has:
- task_id: Unique identifier
- prompt: Function signature and docstring (given to the model)
- canonical_solution: Reference solution
- test: Unit tests to verify correctness
- entry_point: Function name to test

To use the full HumanEval dataset:
    pip install human-eval
    from human_eval.data import read_problems
"""

HUMANEVAL_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": '''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.

    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        "canonical_solution": '''    for i, n1 in enumerate(numbers):
        for j, n2 in enumerate(numbers):
            if i != j and abs(n1 - n2) < threshold:
                return True
    return False
''',
        "test": '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0], 2.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
check(has_close_elements)
''',
        "entry_point": "has_close_elements"
    },
    {
        "task_id": "HumanEval/1",
        "prompt": '''def separate_paren_groups(paren_string: str) -> list[str]:
    """Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those groups into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.

    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        "canonical_solution": '''    result = []
    current = ""
    depth = 0
    for c in paren_string:
        if c == "(":
            depth += 1
            current += c
        elif c == ")":
            depth -= 1
            current += c
            if depth == 0:
                result.append(current)
                current = ""
    return result
''',
        "test": '''
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
    assert candidate('(()(()))') == ['(()(()))']
check(separate_paren_groups)
''',
        "entry_point": "separate_paren_groups"
    },
    {
        "task_id": "HumanEval/2",
        "prompt": '''def truncate_number(number: float) -> float:
    """Given a positive floating point number, it can be decomposed into
    an integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.

    >>> truncate_number(3.5)
    0.5
    """
''',
        "canonical_solution": '''    return number % 1.0
''',
        "test": '''
def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert abs(candidate(123.456) - 0.456) < 1e-6
check(truncate_number)
''',
        "entry_point": "truncate_number"
    },
    {
        "task_id": "HumanEval/3",
        "prompt": '''def below_zero(operations: list[int]) -> bool:
    """You are given a list of deposit and withdrawal operations on a bank account
    that starts with zero balance. Your task is to detect if at any point the balance
    falls below zero, and at that point function should return True. Otherwise return False.

    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
''',
        "canonical_solution": '''    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
''',
        "test": '''
def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True
check(below_zero)
''',
        "entry_point": "below_zero"
    },
    {
        "task_id": "HumanEval/4",
        "prompt": '''def mean_absolute_deviation(numbers: list[float]) -> float:
    """For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |

    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
''',
        "canonical_solution": '''    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
''',
        "test": '''
def check(candidate):
    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6
check(mean_absolute_deviation)
''',
        "entry_point": "mean_absolute_deviation"
    },
    {
        "task_id": "HumanEval/5",
        "prompt": '''def intersperse(numbers: list[int], delimiter: int) -> list[int]:
    """Insert a number 'delimiter' between every two consecutive elements of input list.

    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
''',
        "canonical_solution": '''    if not numbers:
        return []
    result = []
    for n in numbers[:-1]:
        result.append(n)
        result.append(delimiter)
    result.append(numbers[-1])
    return result
''',
        "test": '''
def check(candidate):
    assert candidate([], 7) == []
    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]
    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]
check(intersperse)
''',
        "entry_point": "intersperse"
    },
    {
        "task_id": "HumanEval/6",
        "prompt": '''def parse_nested_parens(paren_string: str) -> list[int]:
    """Input to this function is a string represented multiple groups for nested parentheses
    separated by spaces. For each of the group, output the deepest level of nesting of parentheses.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
''',
        "canonical_solution": '''    def max_depth(s):
        depth = 0
        max_d = 0
        for c in s:
            if c == "(":
                depth += 1
                max_d = max(max_d, depth)
            elif c == ")":
                depth -= 1
        return max_d
    return [max_depth(g) for g in paren_string.split() if g]
''',
        "test": '''
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == [2, 3, 1, 3]
    assert candidate('() (()) ((())) (((())))') == [1, 2, 3, 4]
    assert candidate('(()(())((())))') == [4]
check(parse_nested_parens)
''',
        "entry_point": "parse_nested_parens"
    },
    {
        "task_id": "HumanEval/7",
        "prompt": '''def filter_by_substring(strings: list[str], substring: str) -> list[str]:
    """Filter an input list of strings only for ones that contain given substring.

    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
''',
        "canonical_solution": '''    return [s for s in strings if substring in s]
''',
        "test": '''
def check(candidate):
    assert candidate([], 'john') == []
    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxuj', 'xxx'], 'xxx') == ['xxx', 'xxxuj', 'xxx']
    assert candidate(['xxx', 'asd', 'aaadber', 'john doe', 'xxxuj', 'xxx'], 'john') == ['john doe']
    assert candidate(['grunt', 'hierarchical', 'hierarchical', 'xxx'], 'hierarchical') == ['hierarchical', 'hierarchical']
check(filter_by_substring)
''',
        "entry_point": "filter_by_substring"
    },
    {
        "task_id": "HumanEval/8",
        "prompt": '''def sum_product(numbers: list[int]) -> tuple[int, int]:
    """For a given list of integers, return a tuple consisting of a sum and a product of all integers.
    Empty sum should be equal to 0 and empty product should be equal to 1.

    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
''',
        "canonical_solution": '''    sum_val = 0
    prod_val = 1
    for n in numbers:
        sum_val += n
        prod_val *= n
    return sum_val, prod_val
''',
        "test": '''
def check(candidate):
    assert candidate([]) == (0, 1)
    assert candidate([1, 1, 1]) == (3, 1)
    assert candidate([100, 0]) == (100, 0)
    assert candidate([3, 5, 7]) == (15, 105)
    assert candidate([10]) == (10, 10)
check(sum_product)
''',
        "entry_point": "sum_product"
    },
    {
        "task_id": "HumanEval/9",
        "prompt": '''def rolling_max(numbers: list[int]) -> list[int]:
    """From a given list of integers, generate a list of rolling maximum element
    found until given moment in the sequence.

    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
''',
        "canonical_solution": '''    result = []
    current_max = None
    for n in numbers:
        if current_max is None or n > current_max:
            current_max = n
        result.append(current_max)
    return result
''',
        "test": '''
def check(candidate):
    assert candidate([]) == []
    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4]
    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]
check(rolling_max)
''',
        "entry_point": "rolling_max"
    },
]


def get_problem(task_id: str) -> dict:
    """Get a problem by task_id."""
    for p in HUMANEVAL_PROBLEMS:
        if p["task_id"] == task_id:
            return p
    raise ValueError(f"Problem {task_id} not found")


def get_all_problems() -> list[dict]:
    """Get all problems."""
    return HUMANEVAL_PROBLEMS
