"""
AST-based dynamic rule sanitizer to enforce security constraints on pythonic logical condition rules.
Prevents code injection, arbitrary module access, and un-sanitized dynamic evaluations.
"""

import ast
from typing import Tuple, List, Set

DANGEROUS_FUNCTIONS: Set[str] = {
    "eval", "exec", "__import__", "open", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "system", "popen", "spawn"
}

DANGEROUS_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__bases__", "__mro__", "__globals__", "__builtins__"
}

class ASTRuleSanitizer(ast.NodeVisitor):
    """
    AST Visitor to inspect rule logic AST nodes for prohibited syntax and dangerous calls.
    """

    def __init__(self):
        self.errors: List[str] = []

    def visit_Name(self, node: ast.Name):
        if node.id in DANGEROUS_FUNCTIONS:
            self.errors.append(f"Forbidden function reference detected: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in DANGEROUS_ATTRIBUTES or node.attr.startswith("__"):
            self.errors.append(f"Forbidden attribute access detected: '{node.attr}'")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        self.errors.append("Import statements are strictly prohibited in dynamic rules")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.errors.append("ImportFrom statements are strictly prohibited in dynamic rules")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_FUNCTIONS:
            self.errors.append(f"Prohibited function execution: '{node.func.id}()'")
        self.generic_visit(node)

def sanitize_rule_expression(expr: str) -> Tuple[bool, str]:
    """
    Parse expression using Python AST and return (is_safe, error_message).
    """
    if not expr or not expr.strip():
        return False, "Expression cannot be empty"

    try:
        parsed_ast = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return False, f"AST Syntax Error: {e}"

    sanitizer = ASTRuleSanitizer()
    sanitizer.visit(parsed_ast)

    if sanitizer.errors:
        return False, "; ".join(sanitizer.errors)

    return True, ""
