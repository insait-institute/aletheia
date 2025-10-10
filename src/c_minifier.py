from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CMinifier:
    """
    A modified version of https://github.com/BaseMax/C-Minifier/Minifier.c
    """

    code: str
    cleaned: Optional[str] = field(default=None, init=False)
    minified: Optional[str] = field(default=None, init=False)

    # C keywords used to decide when to keep a space after a token
    C_KEYWORDS = {
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "int",
        "long",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
    }

    @staticmethod
    def _is_ident_char(ch: str) -> bool:
        return ch == "_" or ch.isalnum()

    @staticmethod
    def _skip_whitespace_index(s: str, i: int) -> int:
        n = len(s)
        while i < n and s[i] in " \t\r\n":
            i += 1
        return i

    def remove_comments(self) -> str:
        """Return code with // and /* */ comments removed (strings/chars preserved)."""
        if self.cleaned is not None:
            return self.cleaned

        src = self.code
        out: list[str] = []
        i = 0
        n = len(src)

        in_single = False  # //
        in_multi = False  # /* */
        in_string = False  # "..."
        in_char = False  # '...'
        escaped = False  # inside string/char and last char was '\'

        while i < n:
            c = src[i]

            if in_single:
                # drop until newline, but keep the newline
                if c == "\n":
                    in_single = False
                    out.append(c)
                i += 1
                continue

            if in_multi:
                # drop until */
                if c == "*" and i + 1 < n and src[i + 1] == "/":
                    in_multi = False
                    i += 2
                else:
                    i += 1
                continue

            if in_string:
                out.append(c)
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                i += 1
                continue

            if in_char:
                out.append(c)
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == "'":
                    in_char = False
                i += 1
                continue

            # Not currently inside string/char/comment
            if c == "/" and i + 1 < n:
                nxt = src[i + 1]
                if nxt == "/":
                    in_single = True
                    i += 2
                    continue
                if nxt == "*":
                    in_multi = True
                    i += 2
                    continue

            if c == '"':
                in_string = True
                out.append(c)
                i += 1
                continue

            if c == "'":
                in_char = True
                out.append(c)
                i += 1
                continue

            out.append(c)
            i += 1

        self.cleaned = "".join(out)
        return self.cleaned

    def _parse_char_literal(self, s: str, i: int) -> tuple[str, int]:
        """
        Parse C char literal starting at s[i] == "'".
        Return (replacement_str, index_after_literal).
        Replacement is a decimal string of the character value (like original C port).
        Raises ValueError on malformed literal.
        """
        n = len(s)
        assert s[i] == "'"
        i += 1
        if i >= n:
            raise ValueError("Unterminated char literal")

        if s[i] == "\\":
            i += 1
            if i >= n:
                raise ValueError("Unterminated escape in char literal")
            esc = s[i]

            escapes = {
                "a": ord("\a"),
                "b": ord("\b"),
                "f": ord("\f"),
                "n": ord("\n"),
                "r": ord("\r"),
                "t": ord("\t"),
                "v": ord("\v"),
                "\\": ord("\\"),
                "'": ord("'"),
                '"': ord('"'),
                "?": ord("?"),
                "e": 27,
            }

            if esc in escapes:
                val = escapes[esc]
                i += 1
            elif esc == "x":
                # hex escape: \xHH...
                i += 1
                start = i
                while i < n and (s[i].isdigit() or s[i].lower() in "abcdef"):
                    i += 1
                hexs = s[start:i]
                if not hexs:
                    raise ValueError("Invalid hex escape in char literal")
                val = int(hexs, 16)
            elif esc in "01234567":
                # octal escape (up to 3 digits)
                start = i
                count = 0
                while i < n and s[i] in "01234567" and count < 3:
                    i += 1
                    count += 1
                octs = s[start:i]
                val = int(octs, 8)
            else:
                raise ValueError(f"Unknown escape \\{esc} in char literal")
        else:
            # single character
            val = ord(s[i])
            i += 1

        if i >= n or s[i] != "'":
            raise ValueError("Unterminated char literal")
        i += 1
        return str(val), i

    def minify(self) -> str:
        """Return minified code (requires remove_comments or works on original code)."""
        if self.minified is not None:
            return self.minified

        code = self.cleaned or self.code
        out: list[str] = []
        i = 0
        n = len(code)

        while i < n:
            c = code[i]

            # identifier or keyword
            if self._is_ident_char(c):
                start = i
                while i < n and self._is_ident_char(code[i]):
                    i += 1
                token = code[start:i]
                out.append(token)

                j = self._skip_whitespace_index(code, i)
                next_char = code[j] if j < n else ""

                if token in self.C_KEYWORDS:
                    # keep a space after keyword unless it is immediately followed by '('
                    if next_char and next_char != "(":
                        out.append(" ")
                else:
                    # keep a space if the next visible char starts another identifier
                    if next_char and self._is_ident_char(next_char):
                        out.append(" ")
                continue

            # preprocessor directive — keep entire line, ensure a trailing newline
            if c == "#":
                while i < n and code[i] != "\n":
                    out.append(code[i])
                    i += 1
                out.append("\n")
                if i < n and code[i] == "\n":
                    i += 1
                continue

            # skip plain whitespace
            if c in " \t\r\n":
                i += 1
                continue

            # braces and parentheses
            if c in "{}()":
                out.append(c)
                i += 1
                continue

            # semicolon: append & skip following whitespace
            if c == ";":
                out.append(";")
                i += 1
                while i < n and code[i] in " \t\r\n":
                    i += 1
                continue

            # comma: append & skip following whitespace
            if c == ",":
                out.append(",")
                i += 1
                while i < n and code[i] in " \t\r\n":
                    i += 1
                continue

            # char literal -> convert to decimal representation (safe fallback on error)
            if c == "'":
                try:
                    rep, i = self._parse_char_literal(code, i)
                    out.append(rep)
                except ValueError:
                    # fallback: copy literally until next quote (safer than crashing)
                    out.append("'")
                    i += 1
                    while i < n and code[i] != "'":
                        out.append(code[i])
                        i += 1
                    if i < n:
                        out.append("'")
                        i += 1
                continue

            # string literal -> copy verbatim, preserving escapes
            if c == '"':
                out.append('"')
                i += 1
                while i < n:
                    ch = code[i]
                    out.append(ch)
                    i += 1
                    if ch == "\\" and i < n:
                        # copy escaped char
                        out.append(code[i])
                        i += 1
                    elif ch == '"':
                        break
                continue

            # default: copy single char
            out.append(c)
            i += 1

        self.minified = "".join(out)
        return self.minified

    def process(self) -> str:
        """Run the full pipeline and return the minified result."""
        self.remove_comments()
        return self.minify()
