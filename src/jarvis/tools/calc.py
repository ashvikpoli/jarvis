"""Exact arithmetic. Small models get multiplication wrong; a calculator doesn't.

Evaluates a restricted expression grammar via the AST — no eval(), no names
except whitelisted math functions, so it can't run code.
"""

from __future__ import annotations

import ast
import math
import operator
import re

from pydantic import BaseModel, Field

from jarvis.tools.base import Risk, Tool, ToolContext, ToolError

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 10_000


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("That exponent is too large.")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(f"Unsupported expression: {ast.unparse(node)}")


def calculate(expression: str) -> float | int:
    expr = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # 1,234,567 → 1234567; keeps f(a, b)
    if "%" in expr:  # "15% of 240" style → 0.15*240; plain a % b stays modulo
        expr = expr.replace("% of", "/100*").replace("%of", "/100*")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"I couldn't parse {expression!r} as math.") from e
    try:
        result = _eval(tree)
    except ZeroDivisionError as e:
        raise ToolError("Division by zero.") from e
    except (OverflowError, ValueError) as e:
        raise ToolError(f"Math error: {e}") from e
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return result


class CalcArgs(BaseModel):
    expression: str = Field(
        description="Arithmetic expression, e.g. '1234 * 5678', '(15/100) * 240', 'sqrt(2)'."
    )


class Calculate(Tool):
    name = "calculate"
    description = (
        "Evaluate arithmetic exactly. Use it for EVERY calculation, even simple ones like "
        "15 * 4 — never compute numbers in your head."
    )
    risk = Risk.SAFE
    Args = CalcArgs

    async def run(self, args: CalcArgs, ctx: ToolContext):
        result = calculate(args.expression)
        shown = f"{result:,}" if isinstance(result, int) else f"{result:,.10g}"
        return f"{args.expression} = {shown}"


CALC_TOOLS = [Calculate]
