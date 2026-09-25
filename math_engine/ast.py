"""Mathematical Abstract Syntax Tree (AST) preserving 2D structure.

Preserves:
- Fractions
- Superscripts / Powers
- Subscripts
- Square roots / n-th roots
- Parentheses & Brackets
- Nested operations
- Equations & Inequalities
- Functions

Provides:
- Hierarchical dictionary serialization
- ASCII tree visualization
- Mermaid diagram generation for interactive UI rendering
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class ASTNode(ABC):
    """Base class for all mathematical AST nodes."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize AST node to dictionary."""
        pass

    @abstractmethod
    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        """Render ASCII tree diagram for debugging and terminal display."""
        pass

    @abstractmethod
    def to_latex(self) -> str:
        """Convert AST back to standard LaTeX representation."""
        pass

    def to_mermaid(self) -> str:
        """Generate Mermaid graph syntax for web visualizer."""
        lines = ["graph TD"]
        node_id_counter = [0]

        def _get_children(node: ASTNode) -> list:
            """Return actual child ASTNode objects for any node type."""
            if isinstance(node, (AddNode, MulNode)):
                return list(node.operands)
            if isinstance(node, (SubNode,)):
                return [node.left, node.right]
            if isinstance(node, (EquationNode, InequalityNode)):
                return [node.lhs, node.rhs]
            if isinstance(node, DivNode):
                return [node.numerator, node.denominator]
            if isinstance(node, PowNode):
                return [node.base, node.exponent]
            if isinstance(node, SqrtNode):
                return [node.radicand] + ([node.index] if node.index else [])
            if isinstance(node, FuncNode):
                return [node.argument]
            return []

        def _traverse(node: ASTNode) -> int:
            current_id = node_id_counter[0]
            node_id_counter[0] += 1
            node_name = f"N{current_id}"

            d = node.to_dict()
            label = d.get("label", d.get("type", "Node"))
            safe_label = str(label).replace('"', '\\"')
            lines.append(f'    {node_name}["{safe_label}"]')

            for child in _get_children(node):
                child_id = _traverse(child)
                lines.append(f"    {node_name} --> N{child_id}")

            return current_id

        _traverse(self)
        return "\n".join(lines)



class NumberNode(ASTNode):
    """Represents a numeric constant (integer, rational, or float)."""

    def __init__(self, value: Union[int, float, str]):
        if isinstance(value, str):
            try:
                self.value: Union[int, float] = int(value) if "." not in value else float(value)
            except ValueError:
                self.value = float(value)
        else:
            self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Number", "value": self.value, "label": str(self.value)}

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        return f"{prefix}{connector}Number({self.value})\n"

    def to_latex(self) -> str:
        return str(self.value)


class SymbolNode(ASTNode):
    """Represents a variable or mathematical symbol (e.g. x, y, \\alpha)."""

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Symbol", "name": self.name, "label": self.name}

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        return f"{prefix}{connector}Symbol({self.name})\n"

    def to_latex(self) -> str:
        return self.name


class AddNode(ASTNode):
    """Represents addition of two or more terms."""

    def __init__(self, operands: List[ASTNode]):
        self.operands = operands

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Addition",
            "label": "Addition (+)",
            "children": [op.to_dict() for op in self.operands],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Addition\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.operands):
            res += child.to_ascii_tree(child_prefix, is_last=(i == len(self.operands) - 1))
        return res

    def to_latex(self) -> str:
        return " + ".join(op.to_latex() for op in self.operands)


class SubNode(ASTNode):
    """Represents subtraction: left - right."""

    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Subtraction",
            "label": "Subtraction (-)",
            "children": [self.left.to_dict(), self.right.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Subtraction\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.left.to_ascii_tree(child_prefix, is_last=False)
        res += self.right.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        return f"{self.left.to_latex()} - {self.right.to_latex()}"


class MulNode(ASTNode):
    """Represents multiplication of factors."""

    def __init__(self, operands: List[ASTNode]):
        self.operands = operands

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Multiplication",
            "label": "Multiplication (×)",
            "children": [op.to_dict() for op in self.operands],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Multiplication\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.operands):
            res += child.to_ascii_tree(child_prefix, is_last=(i == len(self.operands) - 1))
        return res

    def to_latex(self) -> str:
        parts = []
        for op in self.operands:
            op_latex = op.to_latex()
            if isinstance(op, (AddNode, SubNode)):
                parts.append(f"({op_latex})")
            else:
                parts.append(op_latex)
        return " \\cdot ".join(parts)


class DivNode(ASTNode):
    """Represents a fraction or division: numerator / denominator."""

    def __init__(self, numerator: ASTNode, denominator: ASTNode):
        self.numerator = numerator
        self.denominator = denominator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Fraction",
            "label": "Fraction (÷)",
            "numerator": self.numerator.to_dict(),
            "denominator": self.denominator.to_dict(),
            "children": [self.numerator.to_dict(), self.denominator.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Fraction\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.numerator.to_ascii_tree(child_prefix, is_last=False)
        res += self.denominator.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        return f"\\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}"


class PowNode(ASTNode):
    """Represents exponentiation: base ^ exponent."""

    def __init__(self, base: ASTNode, exponent: ASTNode):
        self.base = base
        self.exponent = exponent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Power",
            "label": "Power (^)",
            "base": self.base.to_dict(),
            "exponent": self.exponent.to_dict(),
            "children": [self.base.to_dict(), self.exponent.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Power\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.base.to_ascii_tree(child_prefix, is_last=False)
        res += self.exponent.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        base_str = self.base.to_latex()
        if isinstance(self.base, (AddNode, SubNode, MulNode, DivNode)):
            base_str = f"({base_str})"
        return f"{base_str}^{{{self.exponent.to_latex()}}}"


class SqrtNode(ASTNode):
    """Represents a square root or n-th root."""

    def __init__(self, radicand: ASTNode, index: Optional[ASTNode] = None):
        self.radicand = radicand
        self.index = index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SquareRoot",
            "label": "SquareRoot (√)",
            "radicand": self.radicand.to_dict(),
            "children": [self.radicand.to_dict()] if self.index is None else [self.radicand.to_dict(), self.index.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}SquareRoot\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.radicand.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        if self.index is not None:
            return f"\\sqrt[{self.index.to_latex()}]{{{self.radicand.to_latex()}}}"
        return f"\\sqrt{{{self.radicand.to_latex()}}}"


class EquationNode(ASTNode):
    """Represents an equation: LHS = RHS."""

    def __init__(self, lhs: ASTNode, rhs: ASTNode):
        self.lhs = lhs
        self.rhs = rhs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Equation",
            "label": "Equation (=)",
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict(),
            "children": [self.lhs.to_dict(), self.rhs.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Equation\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.lhs.to_ascii_tree(child_prefix, is_last=False)
        res += self.rhs.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        return f"{self.lhs.to_latex()} = {self.rhs.to_latex()}"


class InequalityNode(ASTNode):
    """Represents an inequality: LHS <, <=, >, >= RHS."""

    def __init__(self, lhs: ASTNode, op: str, rhs: ASTNode):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Inequality",
            "label": f"Inequality ({self.op})",
            "operator": self.op,
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict(),
            "children": [self.lhs.to_dict(), self.rhs.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Inequality({self.op})\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.lhs.to_ascii_tree(child_prefix, is_last=False)
        res += self.rhs.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        return f"{self.lhs.to_latex()} {self.op} {self.rhs.to_latex()}"


class FuncNode(ASTNode):
    """Represents a mathematical function call (e.g. sin, cos, ln)."""

    def __init__(self, func_name: str, argument: ASTNode):
        self.func_name = func_name
        self.argument = argument

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Function",
            "label": f"{self.func_name}()",
            "func_name": self.func_name,
            "argument": self.argument.to_dict(),
            "children": [self.argument.to_dict()],
        }

    def to_ascii_tree(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        res = f"{prefix}{connector}Function({self.func_name})\n"
        child_prefix = prefix + ("    " if is_last else "│   ")
        res += self.argument.to_ascii_tree(child_prefix, is_last=True)
        return res

    def to_latex(self) -> str:
        return f"\\{self.func_name}({self.argument.to_latex()})"
