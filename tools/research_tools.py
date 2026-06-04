"""
tools/research_tools.py
---------------------------------------------------------------------
6 Research-Grade Tools:

  1. ArxivSearchTool    - searches real academic papers on arXiv
  2. WikipediaTool      - fetches structured Wikipedia summaries
  3. SymPyTool          - symbolic mathematics (solve, diff, integrate...)
  4. NumericalTool      - numpy/scipy numerical computation
  5. UnitConverterTool  - engineering unit conversions
  6. LatexFormatterTool - converts math expressions to LaTeX
---------------------------------------------------------------------
"""

import math
import re
from .base_tool import BaseTool


# ---------------------------------------------------------------------
# 1. ArXiv Search Tool
# ---------------------------------------------------------------------

class ArxivSearchTool(BaseTool):
    """
    Searches arXiv for academic papers.

    Input format:  <query>
    Example input: GaN HEMT compact model 2DEG charge

    Returns title, authors, year, abstract snippet, and arXiv URL
    for the top N results.
    """

    # Static class variable to track the last request time across all instances
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

    def run(self, input_text: str) -> str:
        import time
        query = input_text.strip()
        max_attempts = 3
        backoff_delay = 3.0

        for attempt in range(max_attempts):
            # Enforce arXiv guideline (minimum 3 seconds spacing between queries)
            current_time = time.time()
            time_since_last = current_time - ArxivSearchTool._last_request_time
            if time_since_last < 3.0:
                time.sleep(3.0 - time_since_last)

            try:
                import arxiv
                client  = arxiv.Client()
                search  = arxiv.Search(
                    query          = query,
                    max_results    = self.max_results,
                    sort_by        = arxiv.SortCriterion.Relevance,
                )
                
                # Fetch results (this triggers the HTTP request)
                papers = list(client.results(search))
                ArxivSearchTool._last_request_time = time.time()
                break  # Successfully fetched
            except ImportError:
                return "ArxivSearchTool requires 'arxiv'. Run: pip install arxiv"
            except Exception as e:
                ArxivSearchTool._last_request_time = time.time()
                err_msg = str(e).lower()
                is_retryable = "503" in err_msg or "429" in err_msg or "too many requests" in err_msg or "service unavailable" in err_msg
                
                if is_retryable and attempt < max_attempts - 1:
                    time.sleep(backoff_delay * (attempt + 1))
                else:
                    return f"arXiv search failed: {e}"

        if not papers:
            return f"No arXiv papers found for: '{query}'"

        lines = [f"arXiv search results for: '{query}'\n"]
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += f" et al."
            year     = paper.published.year if paper.published else "N/A"
            abstract = paper.summary.replace("\n", " ")[:300]
            lines.append(
                f"[{i}] {paper.title}\n"
                f"    Authors : {authors} ({year})\n"
                f"    URL     : {paper.entry_id}\n"
                f"    Abstract: {abstract}...\n"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------
# 2. Wikipedia Tool
# ---------------------------------------------------------------------

class WikipediaTool(BaseTool):
    """
    Fetches a Wikipedia article summary on a topic.

    Input format:  <topic>
    Example input: two-dimensional electron gas

    Returns a structured summary with sections.
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

    def run(self, input_text: str) -> str:
        try:
            import wikipediaapi
            wiki  = wikipediaapi.Wikipedia(
                language   = "en",
                user_agent = "ResearchAgent/1.0"
            )
            topic = input_text.strip()
            page  = wiki.page(topic)

            if not page.exists():
                # Try a fuzzy search fallback
                return (
                    f"Wikipedia page not found for '{topic}'. "
                    f"Try a more specific term or use arxiv_search instead."
                )

            # Split into sentences and take first N
            text      = page.summary
            sentences = text.replace(".\n", ". ").split(". ")
            summary   = ". ".join(sentences[:self.sentences])
            if not summary.endswith("."):
                summary += "."

            # Get top-level section titles
            sections = [s.title for s in page.sections[:6] if s.title]

            output = (
                f"Wikipedia: {page.title}\n"
                f"URL: {page.fullurl}\n\n"
                f"Summary:\n{summary}\n\n"
                f"Key sections in full article: {', '.join(sections)}"
            )
            return output

        except ImportError:
            return "WikipediaTool requires 'wikipedia-api'. Run: pip install wikipedia-api"
        except Exception as e:
            return f"Wikipedia lookup failed for '{input_text}': {e}"


# ---------------------------------------------------------------------
# 3. SymPy Tool - Symbolic Mathematics
# ---------------------------------------------------------------------

class SymPyTool(BaseTool):
    """
    Performs symbolic mathematics using SymPy.

    Input format:  <operation> | <expression> | <variables or limits>
    -----------------------------------------------------------------
    Operations:
      simplify  | expr            | (no var needed)
      expand    | expr            | (no var needed)
      factor    | expr            | (no var needed)
      solve     | equation        | var
      diff      | expr            | var
      diff      | expr            | var, n          (nth derivative)
      integrate | expr            | var             (indefinite)
      integrate | expr            | var, a, b       (definite a->b)
      series    | expr            | var, point, n   (Taylor series)
      limit     | expr            | var, point
      latex     | expr            | (no var needed)
      matrix    | [[a,b],[c,d]]   | (matrix operations)

    Examples:
      simplify  | sin(x)**2 + cos(x)**2 |
      solve     | x**2 - 4*x + 3 | x
      diff      | x**3 * sin(x) | x
      integrate | x**2 * exp(-x) | x, 0, oo
      series    | sin(x) | x, 0, 6
      limit     | sin(x)/x | x, 0
      latex     | Integral(x**2, (x, 0, 1)) |
    """

    def __init__(self):
        super().__init__(
            name="sympy_math",
            description=(
                "Performs symbolic mathematics: simplify, solve equations, "
                "differentiate, integrate, series expansion, limits, and LaTeX output. "
                "Input format: 'operation | expression | variable_or_limits'. "
                "Examples: "
                "'solve | x**2 - 4 | x', "
                "'diff | x**3 * sin(x) | x', "
                "'integrate | exp(-x**2) | x, -oo, oo', "
                "'simplify | sin(x)**2 + cos(x)**2 | ', "
                "'series | ln(1 + exp(x)) | x, 0, 4'. "
                "Use for symbolic derivations, verifying equations, and exact results."
            )
        )

    def run(self, input_text: str) -> str:
        try:
            from sympy import (
                symbols, sympify, simplify, expand, factor, solve,
                diff, integrate, series, limit, latex, Matrix,
                sin, cos, tan, exp, log, sqrt, pi, oo, I, E,
                Symbol, Function, Eq, pretty
            )
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

            transformations = standard_transformations + (implicit_multiplication_application,)

            # -- Parse input ---------------------------------------
            parts = [p.strip() for p in input_text.split("|")]
            if len(parts) < 2:
                return (
                    "Invalid format. Use: 'operation | expression | variable'\n"
                    "Example: 'solve | x**2 - 4 | x'"
                )

            operation = parts[0].lower().strip()
            expr_str  = parts[1].strip()
            var_str   = parts[2].strip() if len(parts) > 2 else ""

            # -- Common symbol namespace ---------------------------
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

            # -- Operations ----------------------------------------

            if operation == "simplify":
                expr   = parse(expr_str)
                result = simplify(expr)
                return f"simplify({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$"

            elif operation == "expand":
                expr   = parse(expr_str)
                result = expand(expr)
                return f"expand({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$"

            elif operation == "factor":
                expr   = parse(expr_str)
                result = factor(expr)
                return f"factor({expr_str})\n= {result}\n\nLaTeX: ${latex(result)}$"

            elif operation == "solve":
                expr = parse(expr_str)
                var  = ns.get(var_str, symbols(var_str)) if var_str else symbols("x")
                # Handle equations (expr = 0 if no "=" in string)
                result = solve(expr, var)
                return (
                    f"solve({expr_str} = 0, {var})\n"
                    f"Solutions: {result}\n"
                    f"LaTeX: ${', '.join(latex(r) for r in result)}$"
                )

            elif operation == "diff":
                expr      = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var       = ns.get(var_parts[0], symbols(var_parts[0]))
                order     = int(var_parts[1]) if len(var_parts) > 1 else 1
                result    = diff(expr, var, order)
                return (
                    f"d{''.join(['^'+str(order) if order>1 else ''])}"
                    f"/d{var}{''.join(['^'+str(order) if order>1 else ''])} "
                    f"({expr_str})\n"
                    f"= {result}\n\nLaTeX: ${latex(result)}$"
                )

            elif operation == "integrate":
                expr      = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var       = ns.get(var_parts[0], symbols(var_parts[0]))

                if len(var_parts) == 3:
                    # Definite integral
                    a      = parse(var_parts[1])
                    b      = parse(var_parts[2])
                    result = integrate(expr, (var, a, b))
                    return (
                        f"Int from {a} to {b} of ({expr_str}) d{var}\n"
                        f"= {simplify(result)}\n\n"
                        f"LaTeX: ${latex(result)}$"
                    )
                else:
                    # Indefinite integral
                    result = integrate(expr, var)
                    return (
                        f"Int ({expr_str}) d{var}\n"
                        f"= {result} + C\n\n"
                        f"LaTeX: ${latex(result)} + C$"
                    )

            elif operation == "series":
                expr      = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var       = ns.get(var_parts[0], symbols(var_parts[0]))
                point     = parse(var_parts[1]) if len(var_parts) > 1 else 0
                n         = int(var_parts[2]) if len(var_parts) > 2 else 6
                result    = series(expr, var, point, n)
                return (
                    f"Taylor series of ({expr_str}) around {var}={point}, "
                    f"up to order {n}:\n"
                    f"= {result}\n\nLaTeX: ${latex(result)}$"
                )

            elif operation == "limit":
                expr      = parse(expr_str)
                var_parts = [v.strip() for v in var_str.split(",")]
                var       = ns.get(var_parts[0], symbols(var_parts[0]))
                point     = parse(var_parts[1]) if len(var_parts) > 1 else 0
                result    = limit(expr, var, point)
                return (
                    f"lim({var} -> {point}) [{expr_str}]\n"
                    f"= {result}\n\nLaTeX: ${latex(result)}$"
                )

            elif operation == "latex":
                expr   = parse(expr_str)
                result = latex(expr)
                return f"LaTeX for '{expr_str}':\n${result}$"

            elif operation == "matrix":
                # Parse matrix from string like [[1,2],[3,4]]
                import ast
                mat_list = ast.literal_eval(expr_str)
                M        = Matrix(mat_list)
                output   = [
                    f"Matrix:\n{M}",
                    f"Determinant: {M.det() if M.is_square else 'N/A (not square)'}",
                ]
                if M.is_square:
                    try:
                        output.append(f"Inverse:\n{M.inv()}")
                        output.append(f"Eigenvalues: {M.eigenvals()}")
                    except Exception:
                        output.append("(Matrix is singular - no inverse)")
                return "\n".join(output)

            else:
                ops = ["simplify", "expand", "factor", "solve", "diff",
                       "integrate", "series", "limit", "latex", "matrix"]
                return f"Unknown operation '{operation}'. Available: {ops}"

        except ImportError:
            return "SymPyTool requires 'sympy'. Run: pip install sympy"
        except Exception as e:
            return f"SymPy error: {e}\nInput was: '{input_text}'"


# ---------------------------------------------------------------------
# 4. Numerical Tool - NumPy / SciPy
# ---------------------------------------------------------------------

class NumericalTool(BaseTool):
    """
    Performs numerical computation using NumPy and SciPy.

    Input format:  <operation> | <parameters>
    -----------------------------------------------------------------
    Operations:
      evaluate   | expression          - evaluate a numeric expression
      matrix     | [[r1],[r2]] | [[b]] - solve Ax = b
      roots      | [a, b, c, ...]      - polynomial roots (coeff high->low)
      stats      | [v1, v2, v3, ...]   - mean, std, min, max of a list
      fft_freq   | sample_rate | n     - FFT frequency bins
      db         | value | ref         - convert to dB (20log10 or 10log10)
      solve_ode  | not supported yet   - use sympy for ODEs

    Examples:
      evaluate | 2 * 3.14159 * 2.4e9 * 50e-12
      matrix   | [[2, 1], [5, 3]] | [4, 7]
      roots    | [1, -6, 11, -6]
      stats    | [1.2, 3.4, 2.1, 5.6, 4.3]
      db       | 0.001 | 1
    """

    def __init__(self):
        super().__init__(
            name="numerical",
            description=(
                "Performs numerical computation with NumPy/SciPy. "
                "Operations: evaluate (numeric expression), matrix (solve Ax=b), "
                "roots (polynomial roots), stats (descriptive statistics), db (dB conversion). "
                "Input format: 'operation | parameters'. "
                "Examples: "
                "'evaluate | 2 * 3.14159 * 2.4e9 * 50e-12', "
                "'matrix | [[2,1],[5,3]] | [4,7]', "
                "'roots | [1, -6, 11, -6]', "
                "'stats | [1.1, 2.2, 3.3, 4.4]'. "
                "Use for numerical answers, matrix equations, and data analysis."
            )
        )

    def run(self, input_text: str) -> str:
        try:
            import numpy as np

            parts     = [p.strip() for p in input_text.split("|")]
            operation = parts[0].lower().strip()

            # Safe namespace for evaluate
            safe_ns = {
                "np": np, "pi": np.pi, "e": np.e,
                "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
                "log10": np.log10, "log2": np.log2,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "abs": np.abs, "ceil": np.ceil, "floor": np.floor,
                "inf": np.inf,
            }

            if operation == "evaluate":
                expr   = parts[1].strip()
                result = eval(expr, {"__builtins__": {}}, safe_ns)
                # Format based on magnitude
                if isinstance(result, (int, float, np.floating)):
                    if abs(result) >= 1e6 or (abs(result) < 1e-3 and result != 0):
                        formatted = f"{result:.6e}"
                    else:
                        formatted = f"{result:.6g}"
                    return f"evaluate: {expr}\n= {formatted}"
                return f"evaluate: {expr}\n= {result}"

            elif operation == "matrix":
                import ast
                A_list = ast.literal_eval(parts[1].strip())
                b_list = ast.literal_eval(parts[2].strip())
                A      = np.array(A_list, dtype=float)
                b      = np.array(b_list, dtype=float)
                x      = np.linalg.solve(A, b)
                return (
                    f"Solve Ax = b:\n"
                    f"A = {A.tolist()}\n"
                    f"b = {b.tolist()}\n"
                    f"Solution x = {[round(v, 6) for v in x.tolist()]}\n"
                    f"Verification Ax = {np.round(A @ x, 6).tolist()}"
                )

            elif operation == "roots":
                import ast
                coeffs = ast.literal_eval(parts[1].strip())
                roots  = np.roots(coeffs)
                formatted = []
                for r in roots:
                    if abs(r.imag) < 1e-10:
                        formatted.append(f"{r.real:.6g}")
                    else:
                        formatted.append(f"{r.real:.4g} + {r.imag:.4g}j")
                return (
                    f"Polynomial roots for coefficients {coeffs}:\n"
                    f"Roots: {formatted}"
                )

            elif operation == "stats":
                import ast
                data   = np.array(ast.literal_eval(parts[1].strip()), dtype=float)
                return (
                    f"Descriptive statistics:\n"
                    f"  Count  : {len(data)}\n"
                    f"  Mean   : {np.mean(data):.6g}\n"
                    f"  Std    : {np.std(data):.6g}\n"
                    f"  Min    : {np.min(data):.6g}\n"
                    f"  Max    : {np.max(data):.6g}\n"
                    f"  Median : {np.median(data):.6g}\n"
                    f"  Sum    : {np.sum(data):.6g}"
                )

            elif operation == "db":
                value = float(eval(parts[1].strip(), {"__builtins__": {}}, safe_ns))
                ref   = float(eval(parts[2].strip(), {"__builtins__": {}}, safe_ns)) if len(parts) > 2 else 1.0
                db_power  = 10 * np.log10(abs(value / ref))
                db_voltage = 20 * np.log10(abs(value / ref))
                return (
                    f"dB conversion: value={value}, ref={ref}\n"
                    f"  Power ratio (10*log10): {db_power:.4f} dB\n"
                    f"  Voltage ratio (20*log10): {db_voltage:.4f} dB"
                )

            else:
                return (f"Unknown operation '{operation}'. "
                        f"Available: evaluate, matrix, roots, stats, db")

        except ImportError:
            return "NumericalTool requires 'numpy'. Run: pip install numpy"
        except Exception as e:
            return f"Numerical error: {e}\nInput: '{input_text}'"


# ---------------------------------------------------------------------
# 5. Unit Converter Tool
# ---------------------------------------------------------------------

class UnitConverterTool(BaseTool):
    """
    Converts between engineering and scientific units.

    Input format:  <value> <from_unit> to <to_unit>
    Examples:
      1 eV to J
      300 K to C
      2.4 GHz to Hz
      0.001 W to dBm
      1 um to nm
    """

    # Conversion table: all values relative to SI base unit
    _UNITS = {
        # Voltage (base: V)
        "V": 1, "mV": 1e-3, "uV": 1e-6, "kV": 1e3, "MV": 1e6,
        # Current (base: A)
        "A": 1, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12,
        # Resistance (base: Ohm)
        "Ohm": 1, "ohm": 1, "mOhm": 1e-3, "kOhm": 1e3, "MOhm": 1e6,
        "mohm": 1e-3, "kohm": 1e3, "Mohm": 1e6,
        # Power (base: W)
        "W": 1, "mW": 1e-3, "uW": 1e-6, "kW": 1e3, "MW": 1e6,
        # Frequency (base: Hz)
        "Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12,
        # Time (base: s)
        "s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15,
        # Length (base: m)
        "m": 1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9,
        "pm": 1e-12, "Angstrom": 1e-10, "A_len": 1e-10,
        # Energy (base: J)
        "J": 1, "mJ": 1e-3, "uJ": 1e-6, "kJ": 1e3, "MJ": 1e6,
        "eV": 1.602176634e-19, "meV": 1.602176634e-22, "keV": 1.602176634e-16,
        # Capacitance (base: F)
        "F": 1, "mF": 1e-3, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12, "fF": 1e-15,
        # Inductance (base: H)
        "H": 1, "mH": 1e-3, "uH": 1e-6, "nH": 1e-9,
        # Pressure (base: Pa)
        "Pa": 1, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
        "bar": 1e5, "atm": 101325, "psi": 6894.757,
        # Charge (base: C)
        "C": 1, "mC": 1e-3, "uC": 1e-6, "nC": 1e-9,
    }

    # Special conversions that need formulas, not just scaling
    _SPECIAL = {
        ("C", "F"):   lambda x: x * 9/5 + 32,
        ("F", "C"):   lambda x: (x - 32) * 5/9,
        ("C", "K"):   lambda x: x + 273.15,
        ("K", "C"):   lambda x: x - 273.15,
        ("F", "K"):   lambda x: (x - 32) * 5/9 + 273.15,
        ("K", "F"):   lambda x: (x - 273.15) * 9/5 + 32,
        ("W",  "dBm"): lambda x: 10 * math.log10(x / 1e-3),
        ("dBm", "W"): lambda x: 1e-3 * 10 ** (x / 10),
        ("W",  "dBW"): lambda x: 10 * math.log10(x),
        ("dBW", "W"): lambda x: 10 ** (x / 10),
    }

    def __init__(self):
        super().__init__(
            name="unit_converter",
            description=(
                "Converts between engineering and scientific units. "
                "Input format: '<value> <from_unit> to <to_unit>'. "
                "Examples: '1 eV to J', '2.4 GHz to Hz', '300 K to C', "
                "'0.001 W to dBm', '100 nm to um', '5 pF to F'. "
                "Supports: voltage, current, resistance, power, frequency, "
                "time, length, energy, capacitance, inductance, pressure, temperature, charge. "
                "Use for unit conversions in engineering calculations."
            )
        )

    def run(self, input_text: str) -> str:
        try:
            text = input_text.strip()

            # Parse: <value> <from_unit> to <to_unit>
            match = re.match(
                r"([+-]?[\d.eE+-]+)\s+(\S+)\s+to\s+(\S+)", text
            )
            if not match:
                return (
                    f"Invalid format. Use: '<value> <from_unit> to <to_unit>'\n"
                    f"Example: '1 eV to J'"
                )

            value     = float(match.group(1))
            from_unit = match.group(2)
            to_unit   = match.group(3)

            # Check special conversions first
            special_key = (from_unit, to_unit)
            if special_key in self._SPECIAL:
                result = self._SPECIAL[special_key](value)
                return (
                    f"{value} {from_unit} = {result:.6g} {to_unit}"
                )

            # Check standard scaling conversions
            if from_unit not in self._UNITS:
                return f"Unknown unit '{from_unit}'. Supported: {list(self._UNITS.keys())}"
            if to_unit not in self._UNITS:
                return f"Unknown unit '{to_unit}'. Supported: {list(self._UNITS.keys())}"

            # Convert via SI base
            si_value = value * self._UNITS[from_unit]
            result   = si_value / self._UNITS[to_unit]

            # Format output nicely
            if abs(result) >= 1e6 or (abs(result) < 1e-4 and result != 0):
                formatted = f"{result:.6e}"
            else:
                formatted = f"{result:.6g}"

            return f"{value} {from_unit} = {formatted} {to_unit}"

        except Exception as e:
            return f"Unit conversion error: {e}\nInput: '{input_text}'"


# ---------------------------------------------------------------------
# 6. LaTeX Formatter Tool
# ---------------------------------------------------------------------

class LatexFormatterTool(BaseTool):
    """
    Formats mathematical expressions and results as LaTeX.

    Input format:  <sympy expression or equation description>
    Returns a clean LaTeX string ready to paste into a document.

    Input examples:
      Integral(exp(-x**2), (x, -oo, oo))
      Matrix([[1, 2], [3, 4]])
      x**2 + 2*x*y + y**2
    """

    def __init__(self):
        super().__init__(
            name="latex_formatter",
            description=(
                "Formats mathematical expressions as LaTeX code. "
                "Input: a SymPy-compatible expression string. "
                "Examples: 'x**2 + y**2', "
                "'Integral(f(x), (x, a, b))', "
                "'Sum(1/n**2, (n, 1, oo))'. "
                "Use this when the final report needs properly formatted equations. "
                "Returns a LaTeX string ready to embed in a document."
            )
        )

    def run(self, input_text: str) -> str:
        try:
            from sympy import symbols, latex, sympify, oo, Function
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations,
                implicit_multiplication_application
            )

            transformations = (
                standard_transformations +
                (implicit_multiplication_application,)
            )

            # Symbol namespace
            x, y, z, t, n, k, a, b = symbols("x y z t n k a b")
            ns = {
                "x": x, "y": y, "z": z, "t": t, "n": n, "k": k,
                "a": a, "b": b, "oo": oo,
                "f": Function("f"), "g": Function("g"),
            }

            expr   = parse_expr(input_text.strip(),
                                local_dict=ns,
                                transformations=transformations)
            result = latex(expr)

            return (
                f"LaTeX output:\n"
                f"  Inline:  ${result}$\n"
                f"  Display: $$\n  {result}\n  $$\n\n"
                f"  Raw LaTeX: {result}"
            )

        except ImportError:
            return "LatexFormatterTool requires 'sympy'. Run: pip install sympy"
        except Exception as e:
            return f"LaTeX formatting error: {e}\nInput: '{input_text}'"