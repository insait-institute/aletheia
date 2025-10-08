# Adopted from https://github.com/CUHK-ARISE/CodeCrash
import random
import re
from typing import List, Tuple

import tree_sitter_cpp as tscpp
import tree_sitter_java as tsj
import tree_sitter_python as tsp
from tree_sitter import Language, Parser

from .misleading_comments import *

LANGS = {
    "python": Language(tsp.language()),
    "java": Language(tsj.language()),
    "cpp": Language(tscpp.language()),
}


# Utility: comment prefix and print snippet by language
def _comment_prefix(lang: str) -> str:
    if lang == "python":
        return "#"
    else:
        # Java & C++ line comment
        return "//"


def _print_line_for(lang: str, text: str) -> str:
    text_esc = text.replace('"', '\\"')
    if lang == "python":
        return f'print("{text_esc}")'
    elif lang == "java":
        return f'System.out.println("{text_esc}");'
    else:  # cpp
        return f'std::cout << "{text_esc}" << std::endl;'


class TreeSitterInserter:
    """
    Walks a tree-sitter parse tree and collects text-edit operations to insert comments or prints.
    Operations are applied to the original source lines (list).
    """

    def __init__(self, code: str, tree, lang: str, once: bool = False, p: float = 1.0, mode: str = "MDC"):
        self.code = code
        self.lines = code.splitlines(True)  # keep newline characters
        self.tree = tree
        self.lang = lang
        self.once = once
        self.p = p
        self.mode = mode  # "MDC" (comments) or "MPS" (prints)
        self.ops: List[Tuple[str, int, str]] = []  # (op_type, row_index, text)
        # track categories to respect `once` flag
        self.contains = {
            "input": False,
            "return": False,
            "variable": False,
            "loop": False,
            "conditional": False,
            "operator": False,
            "operation": False,
        }
        self.initialized_variables = set()

    def should_insert(self, category: str) -> bool:
        if self.once and self.contains.get(category, False):
            return False
        return random.random() <= self.p

    def add_append_comment(self, row: int, comment_text: str):
        """Append an inline comment to existing line (preserves newline)"""
        prefix = _comment_prefix(self.lang)
        # store without modifying now; apply later
        self.ops.append(("append", row, f"{prefix} {comment_text}"))

    def add_insert_line(self, row: int, text_line: str):
        """Insert a new full line at index row (0-based)"""
        # ensure newline at the end
        if not text_line.endswith("\n"):
            text_line = text_line + "\n"
        self.ops.append(("insert", row, text_line))

    def _node_text(self, node):
        return self.code[node.start_byte : node.end_byte].decode("utf8") if isinstance(self.code, bytes) else self.code[node.start_byte : node.end_byte]

    def _extract_method_name_from_call(self, node) -> str:
        # Heuristic: take text of call node and regex for ".name(" or "name("
        call_text = self._node_text(node)
        m = re.search(r"\.([A-Za-z_]\w*)\s*\(", call_text)
        if m:
            return m.group(1)
        m2 = re.search(r"([A-Za-z_]\w*)\s*\(", call_text)
        if m2:
            return m2.group(1)
        return ""

    def visit(self, node):
        t = node.type.lower()

        # FUNCTION / METHOD
        if ("function" in t) or ("method" in t):
            if self.should_insert("input"):
                comment = random.choice(INPUT_PARAMETERS_COMMENT_CANDIDATE)
                # append to the function definition line
                row = node.start_point[0]
                if self.mode == "MDC":
                    self.add_append_comment(row, comment)
                else:  # MPS: insert print after def line with indentation
                    indent = " " * (node.start_point[1] + 4)
                    self.add_insert_line(row + 1, indent + _print_line_for(self.lang, comment))
                self.contains["input"] = True

        # RETURN
        elif "return" in t:
            if self.should_insert("return"):
                useless_value = None
                comment_template = random.choice(RETURN_STATEMENTS_COMMENT_CANDIDATE)
                comment = comment_template.format(useless_value=useless_value)
                row = node.start_point[0]
                if self.mode == "MDC":
                    self.add_append_comment(row, comment)
                else:
                    indent = " " * node.start_point[1]
                    self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                self.contains["return"] = True

        # ASSIGNMENT (heuristic: 'assign' substring)
        elif "assign" in t or "assignment" in t or t == "equal_assignment" or t.endswith("assignment"):
            # try to extract a variable name heuristically from the node text: "x ="
            node_text = self._node_text(node)
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=", node_text)
            var_name = m.group(1) if m else None
            if var_name and var_name not in self.initialized_variables and self.should_insert("variable"):
                self.initialized_variables.add(var_name)
                comment_template = random.choice(VARIABLE_ASSIGNMENTS_COMMENT_CANDIDATE)
                try:
                    comment = comment_template.format(variable=var_name)
                except Exception:
                    comment = comment_template
                row = node.start_point[0]
                if self.mode == "MDC":
                    self.add_append_comment(row, comment)
                else:
                    indent = " " * node.start_point[1]
                    self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                self.contains["variable"] = True
            else:
                # maybe operator-related assignment (e.g., x += y)
                if self.should_insert("operator"):
                    comment = random.choice(OPERATORS_COMMENT_CANDIDATE)
                    row = node.start_point[0]
                    if self.mode == "MDC":
                        self.add_append_comment(row, comment)
                    else:
                        indent = " " * node.start_point[1]
                        self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                    self.contains["operator"] = True

        # AUGMENTED ASSIGN (e.g., +=)
        elif "aug" in t and "assign" in t:
            if self.should_insert("operator"):
                comment = random.choice(OPERATORS_COMMENT_CANDIDATE)
                row = node.start_point[0]
                if self.mode == "MDC":
                    self.add_append_comment(row, comment)
                else:
                    indent = " " * node.start_point[1]
                    self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                self.contains["operator"] = True

        # CALLS / METHOD INVOCATIONS
        elif "call" in t or "invoke" in t or "method" in t and "invocation" in t or "dot" in t:
            method_name = self._extract_method_name_from_call(node)
            if method_name:
                if isinstance(OPERATIONS_COMMENTS_CANDIDATE, dict) and method_name in OPERATIONS_COMMENTS_CANDIDATE and self.should_insert("operation"):
                    comment = random.choice(OPERATIONS_COMMENTS_CANDIDATE[method_name])
                    try:
                        comment = comment.format(name=method_name)
                    except Exception:
                        pass
                    row = node.start_point[0]
                    if self.mode == "MDC":
                        self.add_append_comment(row, comment)
                    else:
                        indent = " " * node.start_point[1]
                        self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                    self.contains["operation"] = True
                elif self.should_insert("operation"):
                    # fallback
                    fallback = OPERATIONS_COMMENTS_CANDIDATE.get("default", None) if isinstance(OPERATIONS_COMMENTS_CANDIDATE, dict) else None
                    comment = random.choice(fallback) if fallback else random.choice(OPERATORS_COMMENT_CANDIDATE)
                    row = node.start_point[0]
                    if self.mode == "MDC":
                        self.add_append_comment(row, comment)
                    else:
                        indent = " " * node.start_point[1]
                        self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                    self.contains["operation"] = True

        # FOR / WHILE (loops)
        elif t.startswith("for") or "for_statement" in t or t.startswith("while") or "while_statement" in t:
            if self.should_insert("loop"):
                comment = random.choice(LOOP_STATEMENTS_COMMENT_CANDIDATE)
                # insert comment as first line in loop body if possible
                # find row for first body child if present else insert at loop start + 1
                row = node.start_point[0] + 1
                indent = " " * (node.start_point[1] + 4)
                if self.mode == "MDC":
                    self.add_insert_line(row, indent + _comment_prefix(self.lang) + " " + comment)
                else:
                    self.add_insert_line(row, indent + _print_line_for(self.lang, comment))
                self.contains["loop"] = True

        # IF / CONDITIONAL
        elif "if" in t and ("if" == t or "if_statement" in t):
            if self.should_insert("conditional"):
                comment = random.choice(CONDITIONAL_STATEMENTS_COMMENT_CANDIDATE)
                row = node.start_point[0]
                if self.mode == "MDC":
                    self.add_append_comment(row, comment)
                else:
                    indent = " " * (node.start_point[1] + 4)
                    self.add_insert_line(row + 1, indent + _print_line_for(self.lang, comment))
                self.contains["conditional"] = True

        # Recurse children
        for child in node.children:
            self.visit(child)

    def apply_ops(self):
        """
        Apply collected ops to self.lines.
        We apply ops in descending row order so insertions do not affect earlier indices.
        """
        # Sort by row descending. For equal rows: inserts before appends? We'll apply inserts first at that row (but descending order ensures stable behavior).
        self.ops.sort(key=lambda x: (x[1], 0 if x[0] == "insert" else 1), reverse=True)
        for op_type, row, text in self.ops:
            # clamp row
            if row < 0:
                row = 0
            if op_type == "append":
                if row >= len(self.lines):
                    # if append location beyond file, append at end
                    self.lines.append(text + "\n")
                else:
                    # preserve original newline
                    original = self.lines[row]
                    self.lines[row] = original.rstrip("\n") + " " + text + "\n"
            elif op_type == "insert":
                if row > len(self.lines):
                    # append at end
                    self.lines.append(text)
                else:
                    self.lines.insert(row, text)

    def result(self) -> str:
        self.apply_ops()
        return "".join(self.lines)


def descriptive_misleading_comments(code: str, lang: str, once: bool = False, p: float = 1.0) -> str:
    """
    High-level wrapper to add misleading comments (MDC) across python/cpp/java using tree-sitter.
    - lang: "python" | "java" | "cpp"
    - once: If True, insert at most one comment of each category
    - p: probability in [0,1] to insert each possible comment
    """
    if lang not in LANGS:
        raise ValueError(f"Unsupported language: {lang}. Supported keys: {list(LANGS.keys())}")
    parser = Parser(LANGS[lang])
    tree = parser.parse(code.encode("utf8"))

    inserter = TreeSitterInserter(code, tree, lang, once=once, p=p, mode="MDC")
    inserter.visit(tree.root_node)
    return inserter.result()


def misleading_prints(code: str, lang: str, once: bool = False, p: float = 1.0) -> str:
    """
    High-level wrapper to insert print statements (MPS) across python/cpp/java using tree-sitter.
    Implementation mirrors process_mdc_treesitter but mode="MPS".
    """
    if lang not in LANGS:
        raise ValueError(f"Unsupported language: {lang}. Supported keys: {list(LANGS.keys())}")
    parser = Parser(LANGS[lang])
    tree = parser.parse(code.encode("utf8"))

    inserter = TreeSitterInserter(code, tree, lang, once=once, p=p, mode="MPS")
    inserter.visit(tree.root_node)
    return inserter.result()


# ---------------------------
# Example usage (quick test)
# ---------------------------
if __name__ == "__main__":
    # Example Python
    sample_py = """
def add(a, b):
    s = a + b
    if s > 10:
        return s
    return 0
"""
    try:
        print("=== Python MDC ===")
        print(misleading_prints(sample_py, "python", once=True, p=0.5))
    except Exception as e:
        print("Error (did you build the languages .so?):", e)

    # Example Java
    sample_java = """
public class Hello {
    public static int add(int a, int b) {
        int s = a + b;
        if (s > 10) {
            return s;
        }
        return 0;
    }
}
"""
    try:
        print("=== Java MDC ===")
        print(misleading_prints(sample_java, "java", once=True, p=1.0))
    except Exception as e:
        print("Error (java):", e)

    # Example C++
    sample_cpp = """
#include <vector>
int add(int a, int b) {
    int s = a + b;
    for (int i = 0; i < 10; ++i) {
        s += i;
    }
    return s;
}
"""
    try:
        print("=== C++ MDC ===")
        print(misleading_prints(sample_cpp, "cpp", once=True, p=1.0))
    except Exception as e:
        print("Error (cpp):", e)
