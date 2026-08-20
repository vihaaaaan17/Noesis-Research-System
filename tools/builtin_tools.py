"""
tools/builtin_tools.py
---------------------------------------------------------------------
4 built-in tools, all free, no extra API keys needed:

  1. CalculatorTool   - safely evaluates math expressions via AST
  2. WebSearchTool    - searches web via DuckDuckGo + Wikipedia fallback
  3. FileReaderTool   - reads text/code files from disk with strict path containment
  4. DateTimeTool     - returns current date, time, timezone

Returns ToolResult contract for clean error handling.
---------------------------------------------------------------------
"""

import ast
import operator
import datetime
import os
from typing import Optional
from .base_tool import BaseTool, ToolResult


# -------------------------------------------------------------
# 1. Calculator Tool
# -------------------------------------------------------------

class CalculatorTool(BaseTool):
    """
    Safely evaluates mathematical expressions using Python's AST parser.
    Supported operators: +, -, *, /, //, %, **, (), negative numbers.
    """

    _OPERATORS = {
        ast.Add:      operator.add,
        ast.Sub:      operator.sub,
        ast.Mult:     operator.mul,
        ast.Div:      operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod:      operator.mod,
        ast.Pow:      operator.pow,
        ast.USub:     operator.neg,
        ast.UAdd:     operator.pos,
    }

    def __init__(self):
        super().__init__(
            name="calculator",
            description=(
                "Evaluates a mathematical expression and returns the result. "
                "Input must be a valid math expression string like '2 + 2', "
                "'(10 * 3) / 4', or '2 ** 8'. "
                "Use this whenever the user asks you to calculate something."
            )
        )

    def run(self, input_text: str) -> ToolResult:
        try:
            expression = input_text.strip()
            tree = ast.parse(expression, mode="eval")
            result = self._eval_node(tree.body)

            if isinstance(result, float) and result.is_integer():
                result = int(result)

            return ToolResult(success=True, output=f"{result}")

        except ZeroDivisionError:
            return ToolResult(success=False, error="Division by zero is not allowed.")
        except Exception as e:
            return ToolResult(success=False, error=f"Error evaluating expression '{input_text}': {e}")

    def _eval_node(self, node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left  = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op    = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op      = self._OPERATORS.get(type(node.op))
            return op(operand)
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")


# -------------------------------------------------------------
# 2. Web Search Tool
# -------------------------------------------------------------

class WebSearchTool(BaseTool):
    """
    Searches the web using DuckDuckGo with Wikipedia fallback.
    Returns bounded formatted results (title, url, snippet).
    """

    def __init__(self, max_results: int = 3, max_retries: int = 2):
        super().__init__(
            name="web_search",
            description=(
                "Searches the web and returns relevant results. "
                "Input should be a clear search query string. "
                "Use this when you need current information or factual lookup."
            )
        )
        self.max_results = max_results
        self.max_retries = max_retries

    def run(self, input_text: str) -> ToolResult:
        try:
            import requests
            import urllib.parse
            from bs4 import BeautifulSoup
            import time

            query = input_text.strip()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://duckduckgo.com/',
            }

            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"

            # Step 1: DuckDuckGo with fast retries
            res = None
            ddg_success = False
            for attempt in range(self.max_retries):
                try:
                    res = requests.get(url, headers=headers, timeout=8)
                    if res.status_code == 200:
                        ddg_success = True
                        break
                    elif res.status_code in (429, 202):
                        if attempt < self.max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                except Exception:
                    if attempt < self.max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue

            if ddg_success and res:
                soup = BeautifulSoup(res.text, 'html.parser')
                divs = soup.find_all('div', class_='result')

                if divs:
                    output_lines = [f"Search results for: '{query}' (via DuckDuckGo)\n"]
                    count = 0
                    for d in divs:
                        if count >= self.max_results:
                            break

                        a_tag = d.find('a', class_='result__a')
                        if not a_tag:
                            continue

                        title = a_tag.text.strip()
                        raw_href = a_tag.get('href')

                        real_url = raw_href
                        if 'uddg=' in raw_href:
                            parsed = urllib.parse.urlparse(raw_href)
                            params = urllib.parse.parse_qs(parsed.query)
                            real_url = params.get('uddg', [raw_href])[0]
                        elif raw_href.startswith('//'):
                            real_url = 'https:' + raw_href

                        snippet_tag = d.find('a', class_='result__snippet')
                        snippet = snippet_tag.text.strip() if snippet_tag else ""
                        if not snippet:
                            snippet_div = d.find('div', class_='result__snippet')
                            snippet = snippet_div.text.strip() if snippet_div else ""

                        count += 1
                        output_lines.append(f"[{count}] {title}")
                        output_lines.append(f"    URL: {real_url}")
                        output_lines.append(f"    {snippet}\n")

                    return ToolResult(success=True, output="\n".join(output_lines))

            # Step 2: Fallback to Wikipedia API
            wiki_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            }
            wiki_headers = {
                "User-Agent": "MASWebSearchAgent/1.0 (contact: user@example.com) requests/2.0"
            }
            try:
                wiki_res = requests.get(wiki_url, params=params, headers=wiki_headers, timeout=8)
                if wiki_res.status_code == 200:
                    data = wiki_res.json()
                    search_results = data.get("query", {}).get("search", [])
                    if search_results:
                        output_lines = [f"Search results for: '{query}' (via Wikipedia fallback)\n"]
                        for i, r in enumerate(search_results[:self.max_results], 1):
                            title = r.get("title")
                            snippet = r.get("snippet", "")
                            clean_snippet = BeautifulSoup(snippet, "html.parser").text
                            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"

                            output_lines.append(f"[{i}] {title}")
                            output_lines.append(f"    URL: {page_url}")
                            output_lines.append(f"    {clean_snippet}\n")
                        return ToolResult(success=True, output="\n".join(output_lines))
                    else:
                        return ToolResult(success=False, error=f"No results found on Wikipedia fallback for: '{query}'")
            except Exception as ex:
                return ToolResult(success=False, error=f"DuckDuckGo rate-limited and Wikipedia fallback error: {ex}")

            return ToolResult(success=False, error=f"Search failed for '{query}': DuckDuckGo rate limit (202/429) and Wikipedia search empty.")
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed for '{input_text}': {e}")


# -------------------------------------------------------------
# 3. File Reader Tool
# -------------------------------------------------------------

class FileReaderTool(BaseTool):
    """
    Reads a text file from disk with strict path containment validation.
    """

    def __init__(self, allowed_dir: str = ".", max_chars: int = 8000):
        super().__init__(
            name="read_file",
            description=(
                "Reads a file from disk and returns its contents as text. "
                "Input must be a valid file path string, e.g. 'notes.txt' or 'data/report.csv'. "
                "Use this when you need to access information stored in a file."
            )
        )
        self.allowed_dir = os.path.abspath(allowed_dir)
        self.max_chars = max_chars

    def run(self, input_text: str) -> ToolResult:
        try:
            file_path = input_text.strip()
            full_path = os.path.abspath(file_path)

            # Strict path containment check using commonpath
            try:
                common = os.path.commonpath([self.allowed_dir, full_path])
                if common != self.allowed_dir:
                    return ToolResult(
                        success=False,
                        error=f"Access denied: '{file_path}' is outside allowed directory '{self.allowed_dir}'"
                    )
            except ValueError:
                # Windows cross-drive access attempt (e.g. C: vs D:)
                return ToolResult(
                    success=False,
                    error=f"Access denied: '{file_path}' is on a different drive than allowed directory '{self.allowed_dir}'"
                )

            if not os.path.exists(full_path):
                return ToolResult(success=False, error=f"File not found: '{file_path}'")

            if not os.path.isfile(full_path):
                return ToolResult(success=False, error=f"'{file_path}' is a directory, not a file.")

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read(self.max_chars)

            full_size = os.path.getsize(full_path)
            truncated = ""
            if full_size > self.max_chars:
                truncated = f"\n\n[Note: File truncated. Showing {self.max_chars} of {full_size} chars]"

            return ToolResult(success=True, output=f"Contents of '{file_path}':\n\n{content}{truncated}")

        except UnicodeDecodeError:
            return ToolResult(success=False, error=f"Could not read '{input_text}': file appears to be binary, not text.")
        except Exception as e:
            return ToolResult(success=False, error=f"Error reading '{input_text}': {e}")


# -------------------------------------------------------------
# 4. DateTime Tool
# -------------------------------------------------------------

class DateTimeTool(BaseTool):
    """
    Returns the current date, time, and day of week.
    """

    def __init__(self):
        super().__init__(
            name="get_datetime",
            description=(
                "Returns the current date and time. "
                "Input can be anything (it is ignored). "
                "Use this whenever the user asks about the current date, time, day of the week, or needs a timestamp."
            )
        )

    def run(self, input_text: str) -> ToolResult:
        now = datetime.datetime.now()
        output = (
            f"Current date and time:\n"
            f"  Date     : {now.strftime('%A, %B %d, %Y')}\n"
            f"  Time     : {now.strftime('%I:%M %p')}\n"
            f"  ISO 8601 : {now.isoformat(timespec='seconds')}\n"
            f"  Weekday  : {now.strftime('%A')}"
        )
        return ToolResult(success=True, output=output)