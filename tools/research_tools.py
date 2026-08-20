"""
tools/research_tools.py
---------------------------------------------------------------------
6 Research-Grade Tools returning ToolResult contracts:

  1. ArxivSearchTool    - searches real academic papers on arXiv
  2. WikipediaTool      - fetches structured Wikipedia summaries
  3. SymPyTool          - symbolic mathematics (solve, diff, integrate...)
  4. NumericalTool      - numpy/scipy numerical computation via safe AST
  5. UnitConverterTool  - engineering unit conversions
  6. LatexFormatterTool - converts math expressions to LaTeX
---------------------------------------------------------------------
"""

import ast
import math
import operator
import re
from typing import Optional
from .base_tool import BaseTool, ToolResult


# ---------------------------------------------------------------------
# 1. ArXiv Search Tool
# ---------------------------------------------------------------------

class ArxivSearchTool(BaseTool):
    """
    Searches arXiv for academic papers.
    Returns title, authors, year, abstract snippet, and arXiv URL.
    """

    _last_request_time = 0.0

    def __init__(self, max_results: int = 4):
        super().__init__(
            name="arxiv_search",
            description=(
                "Searches arXiv for real academic papers on a topic. "
                "Input: a search query string. "
                "Example: 'GaN HEMT compact model threshold voltage'. "
                "Returns paper titles, authors, year, abstract, and URL. "
                "Use this for finding peer-reviewed research, equations from literature, "
                "and state-of-the-art methods."
            )
        )
        self.max_results = max_results

    def run(self, input_text: str) -> ToolResult:
        import time
        query = input_text.strip()
        if not query:
            return ToolResult(success=False, error="Search query cannot be empty.")

        max_attempts = 3
        backoff_delay = 3.0
        papers = []

        for attempt in range(max_attempts):
            current_time = time.time()
            time_since_last = current_time - ArxivSearchTool._last_request_time
            if time_since_last < 3.0:
                time.sleep(3.0 - time_since_last)

            try:
                import arxiv
                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=self.max_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                papers = list(client.results(search))
                ArxivSearchTool._last_request_time = time.time()
                break
            except ImportError:
                return ToolResult(success=False, error="ArxivSearchTool requires 'arxiv'. Run: pip install arxiv")
            except Exception as e:
                ArxivSearchTool._last_request_time = time.time()
                err_msg = str(e).lower()
                is_retryable = "503" in err_msg or "429" in err_msg or "too many requests" in err_msg or "service unavailable" in err_msg

                if is_retryable and attempt < max_attempts - 1:
                    time.sleep(backoff_delay * (attempt + 1))
                else:
                    return ToolResult(success=False, error=f"arXiv search failed: {e}")

        if not papers:
            return ToolResult(success=False, error=f"No arXiv papers found for query: '{query}'")

        lines = [f"arXiv search results for: '{query}'\n"]
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            year = paper.published.year if paper.published else "N/A"
            abstract = paper.summary.replace("\n", " ")[:300]
            lines.append(
                f"[{i}] {paper.title}\n"
                f"    Authors : {authors} ({year})\n"
                f"    URL     : {paper.entry_id}\n"
                f"    Abstract: {abstract}...\n"
            )
        return ToolResult(success=True, output="\n".join(lines))


# ---------------------------------------------------------------------
# 2. Wikipedia Tool
# ---------------------------------------------------------------------

class WikipediaTool(BaseTool):
    """
    Fetches a Wikipedia article summary on a topic.
    """

    def __init__(self, sentences: int = 10):
        super().__init__(
            name="wikipedia",
            description=(
                "Fetches a Wikipedia article summary for a given topic. "
                "Input: a clear topic name. "
                "Example: 'gallium nitride', 'HEMT', 'Fermi level'. "
                "Use this for foundational background, definitions, and "
                "established scientific concepts."
            )
        )
        self.sentences = sentences

    def run(self, input_text: str) -> ToolResult:
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia(
                language="en",
                user_agent="ResearchAgent/1.0"
            )
            topic = input_text.strip()
            if not topic:
                return ToolResult(success=False, error="Wikipedia topic cannot be empty.")

            page = wiki.page(topic)

            if not page.exists():
                return ToolResult(
                    success=False,
                    error=f"Wikipedia page not found for '{topic}'. Try a more specific term or use arxiv_search."
                )

            text = page.summary
            sentences = text.replace(".\n", ". ").split(". ")
            summary = ". ".join(sentences[:self.sentences])
            if not summary.endswith("."):
                summary += "."

            sections = [s.title for s in page.sections[:6] if s.title]

            output = (
                f"Wikipedia: {page.title}\n"
                f"URL: {page.fullurl}\n\n"
                f"Summary:\n{summary}\n\n"
                f"Key sections in full article: {', '.join(sections)}"
            )
            return ToolResult(success=True, output=output)

        except ImportError:
            return ToolResult(success=False, error="WikipediaTool requires 'wikipedia-api'. Run: pip install wikipedia-api")
        except Exception as e:
            return ToolResult(success=False, error=f"Wikipedia lookup failed for '{input_text}': {e}")


# ---------------------------------------------------------------------
# 3. SymPy Tool - Symbolic Mathematics
# ---------------------------------------------------------------------

class SymPyTool(BaseTool):
    """
    Performs symbolic mathematics using SymPy.
    """

    def __init__(self):
        super().__init__(
            name="sympy_math",
            description=(
                "Performs symbolic mathematics: simplify, solve equations, "
                "differentiate, integrate, series expansion, limits, and LaTeX output. "
                "Input format: 'operation | expression | variable_or_limits'. "
                "Examples: 'solve | x**2 - 4 | x', 'diff | x**3 * sin(x) | x'."
            )
        )

    def run(self, input_text: str) -> ToolResult:
        try:
            from sympy import (
                symbols, sympify, simplify, expand, factor, solve,
                diff, integrate, series, limit, latex, Matrix,
                sin, cos, tan, exp, log, sqrt, pi, oo, I, E
            )
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

            transformations = standard_transformations + (implicit_multiplication_application,)

            parts = [p.strip() for p in input_text.split("|")]
            if len(parts) < 2:
                return ToolResult(
                    success=False,
                    error="Invalid format. Use: 'operation | expression | variable'. Example: 'solve | x**2 - 4 | x'"
                )

            operation = parts[0].lower().strip()
            expr_str = parts[1].strip()
            var_str = parts[2].strip() if len(parts) > 2 else ""

            ns = {
                "x": symbols("x"), "y": symbols("y"), "z": symbols("z"),
                "t": symbols("t"), "n": symbols("n"), "k": symbols("k"),
                "a": symbols("a"), "b": symbols("b"), "c": symbols("c"),
                "s": symbols("s"), "w": symbols("w", real=True),
                "pi": pi, "oo": oo, "E": E, "I": I,
                "sin": sin, "cos": cos, "tan": tan,
                "exp": exp, "log": log, "sqrt": sqrt,
            }

            def parse(s):
                return parse_expr(s, local_dict=ns, transformations=transformations)

            if operation == "simplify":
                expr = parse(expr_str)
                result = simplify(expr)
                return ToolResult(success=True, output=f"simplify({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$")

            elif operation == "expand":
                expr = parse(expr_str)
                result = expand(expr)
                return ToolResult(success=True, output=f"expand({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$")

            elif operation == "factor":
                expr = parse(expr_str)
                result = factor(expr)
                return ToolResult(success=True, output=f"factor({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$")

            elif operation == "solve":
                expr = parse(expr_str)
                var = ns.get(var_str, symbols(var_str)) if var_str else symbols("x")
                result = solve(expr, var)
                return ToolResult(
                    success=True,
                    output=f"solve({expr_str} = 0, {var})\nSolutions: {result}\nLaTeX: ${', '.join(latex(r) for r in result)}$"
                )

            elif operation == "diff":
                expr = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var = ns.get(var_parts[0], symbols(var_parts[0]))
                order = int(var_parts[1]) if len(var_parts) > 1 else 1
                result = diff(expr, var, order)
                return ToolResult(
                    success=True,
                    output=f"d^{order}/d{var}^{order} ({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$"
                )

            elif operation == "integrate":
                expr = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var = ns.get(var_parts[0], symbols(var_parts[0]))

                if len(var_parts) == 3:
                    a = parse(var_parts[1])
                    b = parse(var_parts[2])
                    result = integrate(expr, (var, a, b))
                    return ToolResult(
                        success=True,
                        output=f"Int from {a} to {b} of ({expr_str}) d{var}\n= {simplify(result)}\n\nLaTeX: ${latex(result)}$"
                    )
                else:
                    result = integrate(expr, var)
                    return ToolResult(
                        success=True,
                        output=f"Int ({expr_str}) d{var}\n= {result} + C\n\nLaTeX: ${latex(result)} + C$"
                    )

            elif operation == "series":
                expr = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var = ns.get(var_parts[0], symbols(var_parts[0]))
                point = parse(var_parts[1]) if len(var_parts) > 1 else 0
                n = int(var_parts[2]) if len(var_parts) > 2 else 6
                result = series(expr, var, point, n)
                return ToolResult(
                    success=True,
                    output=f"Taylor series of ({expr_str}) around {var}={point}, up to order {n}:\n= {result}\n\nLaTeX: ${latex(result)}$"
                )

            elif operation == "limit":
                expr = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var = ns.get(var_parts[0], symbols(var_parts[0]))
                point = parse(var_parts[1]) if len(var_parts) > 1 else 0
                result = limit(expr, var, point)
                return ToolResult(
                    success=True,
                    output=f"lim({var} -> {point}) [{expr_str}]\n= {result}\n\nLaTeX: ${latex(result)}$"
                )

            elif operation == "latex":
                expr = parse(expr_str)
                result = latex(expr)
                return ToolResult(success=True, output=f"LaTeX for '{expr_str}':\n${result}$")

            elif operation == "matrix":
                mat_list = ast.literal_eval(expr_str)
                M = Matrix(mat_list)
                output = [
                    f"Matrix:\n{M}",
                    f"Determinant: {M.det() if M.is_square else 'N/A (not square)'}",
                ]
                if M.is_square:
                    try:
                        output.append(f"Inverse:\n{M.inv()}")
                        output.append(f"Eigenvalues: {M.eigenvals()}")
                    except Exception:
                        output.append("(Matrix is singular - no inverse)")
                return ToolResult(success=True, output="\n".join(output))

            else:
                ops = ["simplify", "expand", "factor", "solve", "diff", "integrate", "series", "limit", "latex", "matrix"]
                return ToolResult(success=False, error=f"Unknown operation '{operation}'. Available: {ops}")

        except ImportError:
            return ToolResult(success=False, error="SymPyTool requires 'sympy'. Run: pip install sympy")
        except Exception as e:
            return ToolResult(success=False, error=f"SymPy error: {e} (Input was: '{input_text}')")


# ---------------------------------------------------------------------
# 4. Numerical Tool - Safe AST Evaluator
# ---------------------------------------------------------------------

class NumericalTool(BaseTool):
    """
    Performs numerical computation using NumPy and SciPy with safe AST node evaluation.
    """

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def __init__(self):
        super().__init__(
            name="numerical",
            description=(
                "Performs numerical computation with NumPy/SciPy. "
                "Operations: evaluate (numeric expression), matrix (solve Ax=b), "
                "roots (polynomial roots), stats (descriptive statistics), db (dB conversion). "
                "Input format: 'operation | parameters'."
            )
        )

    def _eval_ast_node(self, node, ns):
        """Safely evaluate AST node without unrestricted eval()."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in ns:
                return ns[node.id]
            raise ValueError(f"Undefined variable in expression: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_node(node.left, ns)
            right = self._eval_ast_node(node.right, ns)
            op = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_node(node.operand, ns)
            op = self._OPERATORS.get(type(node.op))
            return op(operand)
        elif isinstance(node, ast.Call):
            func = self._eval_ast_node(node.func, ns)
            args = [self._eval_ast_node(arg, ns) for arg in node.args]
            if callable(func):
                return func(*args)
            raise ValueError(f"Function {node.func} is not callable.")
        elif isinstance(node, ast.Attribute):
            val = self._eval_ast_node(node.value, ns)
            return getattr(val, node.attr)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _safe_eval(self, expr_str: str, ns: dict):
        """Parse and safely evaluate numeric expression."""
        tree = ast.parse(expr_str.strip(), mode="eval")
        return self._eval_ast_node(tree.body, ns)

    def run(self, input_text: str) -> ToolResult:
        try:
            import numpy as np

            parts = [p.strip() for p in input_text.split("|")]
            if not parts or not parts[0]:
                return ToolResult(success=False, error="Numerical operation cannot be empty.")

            operation = parts[0].lower().strip()

            safe_ns = {
                "np": np, "pi": np.pi, "e": np.e,
                "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
                "log10": np.log10, "log2": np.log2,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "abs": np.abs, "ceil": np.ceil, "floor": np.floor,
                "inf": np.inf,
            }

            if operation == "evaluate":
                if len(parts) < 2:
                    return ToolResult(success=False, error="Missing expression for evaluate.")
                expr = parts[1].strip()
                result = self._safe_eval(expr, safe_ns)
                if isinstance(result, (int, float, np.floating)):
                    if abs(result) >= 1e6 or (abs(result) < 1e-3 and result != 0):
                        formatted = f"{result:.6e}"
                    else:
                        formatted = f"{result:.6g}"
                    return ToolResult(success=True, output=f"evaluate: {expr}\n= {formatted}")
                return ToolResult(success=True, output=f"evaluate: {expr}\n= {result}")

            elif operation == "matrix":
                if len(parts) < 3:
                    return ToolResult(success=False, error="Matrix operation requires format: 'matrix | [[A]] | [b]'")
                A_list = ast.literal_eval(parts[1].strip())
                b_list = ast.literal_eval(parts[2].strip())
                A = np.array(A_list, dtype=float)
                b = np.array(b_list, dtype=float)
                x = np.linalg.solve(A, b)
                return ToolResult(
                    success=True,
                    output=(
                        f"Solve Ax = b:\nA = {A.tolist()}\nb = {b.tolist()}\n"
                        f"Solution x = {[round(v, 6) for v in x.tolist()]}\n"
                        f"Verification Ax = {np.round(A @ x, 6).tolist()}"
                    )
                )

            elif operation == "roots":
                if len(parts) < 2:
                    return ToolResult(success=False, error="Roots operation requires format: 'roots | [coeffs]'")
                coeffs = ast.literal_eval(parts[1].strip())
                roots = np.roots(coeffs)
                formatted = []
                for r in roots:
                    if abs(r.imag) < 1e-10:
                        formatted.append(f"{r.real:.6g}")
                    else:
                        formatted.append(f"{r.real:.4g} + {r.imag:.4g}j")
                return ToolResult(
                    success=True,
                    output=f"Polynomial roots for coefficients {coeffs}:\nRoots: {formatted}"
                )

            elif operation == "stats":
                if len(parts) < 2:
                    return ToolResult(success=False, error="Stats operation requires format: 'stats | [data]'")
                data = np.array(ast.literal_eval(parts[1].strip()), dtype=float)
                return ToolResult(
                    success=True,
                    output=(
                        f"Descriptive statistics:\n  Count  : {len(data)}\n"
                        f"  Mean   : {np.mean(data):.6g}\n  Std    : {np.std(data):.6g}\n"
                        f"  Min    : {np.min(data):.6g}\n  Max    : {np.max(data):.6g}\n"
                        f"  Median : {np.median(data):.6g}\n  Sum    : {np.sum(data):.6g}"
                    )
                )

            elif operation == "db":
                if len(parts) < 2:
                    return ToolResult(success=False, error="dB operation requires format: 'db | value | ref'")
                value = float(self._safe_eval(parts[1].strip(), safe_ns))
                ref = float(self._safe_eval(parts[2].strip(), safe_ns)) if len(parts) > 2 else 1.0
                db_power = 10 * np.log10(abs(value / ref))
                db_voltage = 20 * np.log10(abs(value / ref))
                return ToolResult(
                    success=True,
                    output=(
                        f"dB conversion: value={value}, ref={ref}\n"
                        f"  Power ratio (10*log10): {db_power:.4f} dB\n"
                        f"  Voltage ratio (20*log10): {db_voltage:.4f} dB"
                    )
                )

            else:
                return ToolResult(success=False, error=f"Unknown operation '{operation}'. Available: evaluate, matrix, roots, stats, db")

        except ImportError:
            return ToolResult(success=False, error="NumericalTool requires 'numpy'. Run: pip install numpy")
        except Exception as e:
            return ToolResult(success=False, error=f"Numerical error: {e} (Input: '{input_text}')")


# ---------------------------------------------------------------------
# 5. Unit Converter Tool
# ---------------------------------------------------------------------

class UnitConverterTool(BaseTool):
    """
    Converts between engineering and scientific units.
    """

    _UNITS = {
        "V": 1, "mV": 1e-3, "uV": 1e-6, "kV": 1e3, "MV": 1e6,
        "A": 1, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12,
        "Ohm": 1, "ohm": 1, "mOhm": 1e-3, "kOhm": 1e3, "MOhm": 1e6, "mohm": 1e-3, "kohm": 1e3, "Mohm": 1e6,
        "W": 1, "mW": 1e-3, "uW": 1e-6, "kW": 1e3, "MW": 1e6,
        "Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12,
        "s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15,
        "m": 1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12, "Angstrom": 1e-10, "A_len": 1e-10,
        "J": 1, "mJ": 1e-3, "uJ": 1e-6, "kJ": 1e3, "MJ": 1e6, "eV": 1.602176634e-19, "meV": 1.602176634e-22, "keV": 1.602176634e-16,
        "F": 1, "mF": 1e-3, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12, "fF": 1e-15,
        "H": 1, "mH": 1e-3, "uH": 1e-6, "nH": 1e-9,
        "Pa": 1, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "atm": 101325, "psi": 6894.757,
        "C": 1, "mC": 1e-3, "uC": 1e-6, "nC": 1e-9,
    }

    _SPECIAL = {
        ("C", "F"): lambda x: x * 9/5 + 32,
        ("F", "C"): lambda x: (x - 32) * 5/9,
        ("C", "K"): lambda x: x + 273.15,
        ("K", "C"): lambda x: x - 273.15,
        ("F", "K"): lambda x: (x - 32) * 5/9 + 273.15,
        ("K", "F"): lambda x: (x - 273.15) * 9/5 + 32,
        ("W", "dBm"): lambda x: 10 * math.log10(x / 1e-3),
        ("dBm", "W"): lambda x: 1e-3 * 10 ** (x / 10),
        ("W", "dBW"): lambda x: 10 * math.log10(x),
        ("dBW", "W"): lambda x: 10 ** (x / 10),
    }

    def __init__(self):
        super().__init__(
            name="unit_converter",
            description=(
                "Converts between engineering and scientific units. "
                "Input format: '<value> <from_unit> to <to_unit>'. "
                "Examples: '1 eV to J', '2.4 GHz to Hz', '300 K to C'."
            )
        )

    def run(self, input_text: str) -> ToolResult:
        try:
            text = input_text.strip()
            match = re.match(r"([+-]?[\d.eE+-]+)\s+(\S+)\s+to\s+(\S+)", text)
            if not match:
                return ToolResult(
                    success=False,
                    error="Invalid format. Use: '<value> <from_unit> to <to_unit>'. Example: '1 eV to J'"
                )

            value = float(match.group(1))
            from_unit = match.group(2)
            to_unit = match.group(3)

            special_key = (from_unit, to_unit)
            if special_key in self._SPECIAL:
                result = self._SPECIAL[special_key](value)
                return ToolResult(success=True, output=f"{value} {from_unit} = {result:.6g} {to_unit}")

            if from_unit not in self._UNITS:
                return ToolResult(success=False, error=f"Unknown unit '{from_unit}'. Supported: {list(self._UNITS.keys())}")
            if to_unit not in self._UNITS:
                return ToolResult(success=False, error=f"Unknown unit '{to_unit}'. Supported: {list(self._UNITS.keys())}")

            si_value = value * self._UNITS[from_unit]
            result = si_value / self._UNITS[to_unit]

            if abs(result) >= 1e6 or (abs(result) < 1e-4 and result != 0):
                formatted = f"{result:.6e}"
            else:
                formatted = f"{result:.6g}"

            return ToolResult(success=True, output=f"{value} {from_unit} = {formatted} {to_unit}")

        except Exception as e:
            return ToolResult(success=False, error=f"Unit conversion error: {e} (Input: '{input_text}')")


# ---------------------------------------------------------------------
# 6. LaTeX Formatter Tool
# ---------------------------------------------------------------------

class LatexFormatterTool(BaseTool):
    """
    Formats mathematical expressions and results as LaTeX.
    """

    def __init__(self):
        super().__init__(
            name="latex_formatter",
            description=(
                "Formats mathematical expressions as LaTeX code. "
                "Input: a SymPy-compatible expression string. "
                "Examples: 'x**2 + y**2', 'Integral(f(x), (x, a, b))'."
            )
        )

    def run(self, input_text: str) -> ToolResult:
        try:
            from sympy import symbols, latex, sympify, oo, Function
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

            transformations = standard_transformations + (implicit_multiplication_application,)

            x, y, z, t, n, k, a, b = symbols("x y z t n k a b")
            ns = {
                "x": x, "y": y, "z": z, "t": t, "n": n, "k": k,
                "a": a, "b": b, "oo": oo,
                "f": Function("f"), "g": Function("g"),
            }

            expr = parse_expr(input_text.strip(), local_dict=ns, transformations=transformations)
            result = latex(expr)

            return ToolResult(
                success=True,
                output=f"LaTeX output:\n  Inline:  ${result}$\n  Display: $$\n  {result}\n  $$\n\n  Raw LaTeX: {result}"
            )

        except ImportError:
            return ToolResult(success=False, error="LatexFormatterTool requires 'sympy'. Run: pip install sympy")
        except Exception as e:
            return ToolResult(success=False, error=f"LaTeX formatting error: {e} (Input: '{input_text}')")