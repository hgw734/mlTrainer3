import ast
import os
import sys
import traceback
import tokenize
import io
from lib2to3.refactor import RefactoringTool, get_fixers_from_package

LOG_FILE = "syntax_fix_log.txt"

COMMON_FIXES = {
    'EOL while scanning string literal': lambda line: line + '"',
    'unexpected EOF while parsing': lambda line: line + ')',
    'missing )': lambda line: line + ')',
    'invalid syntax': lambda line: line + '  # fixme: check manually',
}


def fix_line(line, error_msg):
    for key in COMMON_FIXES:
        if key in error_msg:
            return COMMON_FIXES[key](line)
            return line

            def apply_lib2to3(source_code):
                fixers = get_fixers_from_package("lib2to3.fixes")
                tool = RefactoringTool(fixers)
                try:
                    return str(tool.refactor_string(source_code, name="temp"))
                    except Exception as e:
                        return source_code

                        def fix_indentation(code):
                            lines = code.split("\n")
                            fixed = []
                            indent = 0
                            for line in lines:
                                stripped = line.strip()
                                if not stripped:
                                    fixed.append("")
                                    continue
                                if stripped.endswith(":"):
                                    fixed.append("    " * indent + stripped)
                                    indent += 1
                                    elif stripped in ("return", "break", "pass", "continue") and indent > 0:
                                        fixed.append(
                                            "    " * indent + stripped)
                                        indent -= 1
                                        else:
                                            fixed.append(
                                                "    " * indent + stripped)
                                            return "\n".join(fixed)

                                            def try_ast_parse(code):
                                                try:
                                                    ast.parse(code)
                                                    return True, None
                                                    except SyntaxError as e:
                                                        return False, e

                                                        def fix_file(filepath):
                                                            with open(filepath, "r", encoding="utf-8") as f:
                                                                lines = f.readlines()

                                                                fixed_lines = []
                                                                error_log = []

                                                                for i, line in enumerate(
                                                                        lines):
                                                                    test_code = "".join(
                                                                        fixed_lines + [line] + lines[i + 1:])
                                                                    ok, err = try_ast_parse(
                                                                        test_code)
                                                                    if ok:
                                                                        fixed_lines.append(
                                                                            line)
                                                                        else:
                                                                            msg = str(
                                                                                err)
                                                                            fixed = fix_line(
                                                                                line, msg)
                                                                            fixed_lines.append(
                                                                                fixed)
                                                                            error_log.append(
                                                                                (i + 1, msg, line.strip()))

                                                                            final_code = "".join(
                                                                                fixed_lines)
                                                                            final_code = apply_lib2to3(
                                                                                final_code)
                                                                            final_code = fix_indentation(
                                                                                final_code)

                                                                            # Final
                                                                            # check
                                                                            ok, err = try_ast_parse(
                                                                                final_code)
                                                                            if not ok:
                                                                                error_log.append(
                                                                                    ("FINAL", str(err), ""))

                                                                                with open(filepath, "w", encoding="utf-8") as f:
                                                                                    f.write(
                                                                                        final_code)

                                                                                    return error_log

                                                                                    def process_path(
                                                                                            path):
                                                                                        all_logs = []
                                                                                        if os.path.isfile(
                                                                                                path) and path.endswith(".py"):
                                                                                            print(
                                                                                                f"🔧 Fixing: {path}")
                                                                                            errors = fix_file(
                                                                                                path)
                                                                                            all_logs.extend(
                                                                                                [(path, *e) for e in errors])
                                                                                            elif os.path.isdir(path):
                                                                                                for root, _, files in os.walk(
                                                                                                        path):
                                                                                                    for file in files:
                                                                                                        if file.endswith(
                                                                                                                ".py"):
                                                                                                            full_path = os.path.join(
                                                                                                                root, file)
                                                                                                            print(
                                                                                                                f"🔧 Fixing: {full_path}")
                                                                                                            errors = fix_file(
                                                                                                                full_path)
                                                                                                            all_logs.extend(
                                                                                                                [(full_path, *e) for e in errors])
                                                                                                            else:
                                                                                                                print(
                                                                                                                    "❌ Invalid path")
                                                                                                                return

                                                                                                            # Log
                                                                                                            with open(LOG_FILE, "w", encoding="utf-8") as logf:
                                                                                                                for item in all_logs:
                                                                                                                    logf.write(
                                                                                                                        f"{item[0]}: Line {item[1]} - {item[2]} :: {item[3]}\n")

                                                                                                                    print(
                                                                                                                        f"\n✅ Done. Log saved to {LOG_FILE}")

                                                                                                                    if __name__ == "__main__":
                                                                                                                        if len(
                                                                                                                                sys.argv) != 2:
                                                                                                                            print(
                                                                                                                                "Usage: python fix_syntax_errors.py <file_or_directory>")
                                                                                                                            sys.exit(
                                                                                                                                1)
                                                                                                                            path = sys.argv[
                                                                                                                                1]
                                                                                                                            process_path(
                                                                                                                                path)
