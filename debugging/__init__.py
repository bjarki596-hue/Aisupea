"""
Aisupea Debugging System

Self-debugging module that parses Python tracebacks, detects missing imports,
and proposes fixes automatically.
"""

import traceback
import sys
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class ErrorAnalyzer:
    """Analyzes Python errors and tracebacks."""

    def __init__(self):
        self.error_patterns = {
            'ImportError': self._analyze_import_error,
            'ModuleNotFoundError': self._analyze_module_error,
            'NameError': self._analyze_name_error,
            'AttributeError': self._analyze_attribute_error,
            'SyntaxError': self._analyze_syntax_error,
            'IndentationError': self._analyze_indentation_error,
            'TypeError': self._analyze_type_error,
            'ValueError': self._analyze_value_error,
        }

    def analyze_error(self, error: Exception, traceback_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a Python error and provide debugging information.

        Args:
            error: The exception object
            traceback_str: String representation of traceback

        Returns:
            Analysis results with suggested fixes
        """
        error_type = type(error).__name__

        if traceback_str is None:
            traceback_str = traceback.format_exc()

        analysis = {
            'error_type': error_type,
            'error_message': str(error),
            'traceback': traceback_str,
            'file': None,
            'line': None,
            'suggestions': [],
            'fixes': []
        }

        # Extract file and line information
        file_info = self._extract_file_info(traceback_str)
        if file_info:
            analysis.update(file_info)

        # Analyze specific error type
        analyzer = self.error_patterns.get(error_type)
        if analyzer:
            specific_analysis = analyzer(error, traceback_str)
            analysis.update(specific_analysis)

        return analysis

    def _extract_file_info(self, traceback_str: str) -> Optional[Dict[str, Any]]:
        """Extract file and line information from traceback."""
        lines = traceback_str.split('\n')

        for line in lines:
            # Look for file and line information
            match = re.search(r'File "([^"]+)", line (\d+)', line)
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))

                # Try to extract the problematic line
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_lines = f.readlines()
                        if line_num <= len(file_lines):
                            problematic_line = file_lines[line_num - 1].strip()
                        else:
                            problematic_line = None
                except:
                    problematic_line = None

                return {
                    'file': file_path,
                    'line': line_num,
                    'problematic_line': problematic_line
                }

        return None

    def _analyze_import_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze ImportError."""
        error_msg = str(error)
        suggestions = []
        fixes = []

        # Check for common import issues
        if 'No module named' in error_msg:
            module_name = error_msg.split("'")[-2] if "'" in error_msg else ""

            suggestions.append(f"Module '{module_name}' is not installed or not in Python path")
            suggestions.append("Try installing with: pip install " + module_name)

            # Check if it's a common misspelling
            common_modules = {
                'numpy': ['numPy', 'Numpy', 'np'],
                'pandas': ['Pandas', 'pd'],
                'matplotlib': ['Matplotlib', 'mpl'],
                'tensorflow': ['Tensorflow', 'tf'],
                'torch': ['PyTorch', 'pytorch'],
            }

            for correct, misspellings in common_modules.items():
                if module_name in misspellings:
                    suggestions.append(f"Did you mean '{correct}' instead of '{module_name}'?")
                    fixes.append({
                        'type': 'replace_import',
                        'old': module_name,
                        'new': correct
                    })

        return {'suggestions': suggestions, 'fixes': fixes}

    def _analyze_module_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze ModuleNotFoundError (similar to ImportError)."""
        return self._analyze_import_error(error, traceback_str)

    def _analyze_name_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze NameError."""
        error_msg = str(error)
        suggestions = []
        fixes = []

        if "name '" in error_msg and "' is not defined" in error_msg:
            var_name = error_msg.split("'")[1]

            suggestions.append(f"Variable '{var_name}' is not defined in the current scope")
            suggestions.append("Check if the variable is spelled correctly")
            suggestions.append("Make sure the variable is assigned before use")
            suggestions.append("Check if you need to import something")

            # Check for common undefined names
            common_fixes = {
                'np': 'import numpy as np',
                'pd': 'import pandas as pd',
                'plt': 'import matplotlib.pyplot as plt',
                'tf': 'import tensorflow as tf',
                'torch': 'import torch',
            }

            if var_name in common_fixes:
                suggestions.append(f"Try adding: {common_fixes[var_name]}")
                fixes.append({
                    'type': 'add_import',
                    'import_statement': common_fixes[var_name]
                })

        return {'suggestions': suggestions, 'fixes': fixes}

    def _analyze_attribute_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze AttributeError."""
        error_msg = str(error)
        suggestions = []
        fixes = []

        if "object has no attribute" in error_msg:
            parts = error_msg.split("'")
            if len(parts) >= 4:
                obj_type = parts[1]
                attr_name = parts[3]

                suggestions.append(f"Object of type '{obj_type}' has no attribute '{attr_name}'")
                suggestions.append("Check the object type and available methods")
                suggestions.append("Check for typos in attribute name")

        return {'suggestions': suggestions, 'fixes': fixes}

    def _analyze_syntax_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze SyntaxError."""
        suggestions = []
        fixes = []

        suggestions.append("Check for syntax errors like missing colons, parentheses, or quotes")
        suggestions.append("Check indentation")
        suggestions.append("Check for invalid Python syntax")

        return {'suggestions': suggestions, 'fixes': fixes}

    def _analyze_indentation_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze IndentationError."""
        suggestions = []
        fixes = []

        suggestions.append("Check that indentation is consistent (use spaces or tabs, not both)")
        suggestions.append("Check that code blocks are properly indented")
        suggestions.append("Python uses indentation to define code blocks")

        return {'suggestions': suggestions, 'fixes': fixes}

    def _analyze_type_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze TypeError."""
        error_msg = str(error)
        suggestions = []
        fixes = []

        suggestions.append("Check that you're using the correct types for operations")
        suggestions.append("Check function signatures and argument types")

        if "unsupported operand type" in error_msg:
            suggestions.append("You're trying to use an operator on incompatible types")

        return {'suggestions': suggestions, 'fixes': fixes}

    def _analyze_value_error(self, error: Exception, traceback_str: str) -> Dict[str, Any]:
        """Analyze ValueError."""
        suggestions = []
        fixes = []

        suggestions.append("Check that function arguments have valid values")
        suggestions.append("Check data types and ranges")
        suggestions.append("Check array shapes and dimensions")

        return {'suggestions': suggestions, 'fixes': fixes}


class CodeFixer:
    """Applies automatic fixes to code."""

    def __init__(self):
        self.fixers = {
            'add_import': self._fix_add_import,
            'replace_import': self._fix_replace_import,
        }

    def apply_fix(self, file_path: str, fix: Dict[str, Any]) -> bool:
        """
        Apply a fix to a file.

        Args:
            file_path: Path to the file to fix
            fix: Fix specification

        Returns:
            True if fix was applied successfully
        """
        fixer = self.fixers.get(fix['type'])
        if not fixer:
            return False

        try:
            return fixer(file_path, fix)
        except Exception:
            return False

    def _fix_add_import(self, file_path: str, fix: Dict[str, Any]) -> bool:
        """Add an import statement to a file."""
        import_statement = fix['import_statement']

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        # Find where to insert the import
        insert_idx = 0

        # Skip shebang
        if lines and lines[0].startswith('#!'):
            insert_idx = 1

        # Skip existing imports and blank lines
        while insert_idx < len(lines):
            line = lines[insert_idx].strip()
            if line.startswith(('import ', 'from ')) or line == '':
                insert_idx += 1
            else:
                break

        # Insert the import
        lines.insert(insert_idx, import_statement)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return True

    def _fix_replace_import(self, file_path: str, fix: Dict[str, Any]) -> bool:
        """Replace an import statement in a file."""
        old_module = fix['old']
        new_module = fix['new']

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace import statements
        import_pattern = rf'\b{re.escape(old_module)}\b'
        new_content = re.sub(import_pattern, new_module, content)

        # Write back if changed
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True

        return False


class Debugger:
    """Main debugging system that coordinates error analysis and fixing."""

    def __init__(self):
        self.error_analyzer = ErrorAnalyzer()
        self.code_fixer = CodeFixer()

    def debug_error(self, error: Exception, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Debug a Python error and optionally apply fixes.

        Args:
            error: The exception that occurred
            auto_fix: Whether to automatically apply safe fixes

        Returns:
            Debugging results
        """
        analysis = self.error_analyzer.analyze_error(error)

        results = {
            'analysis': analysis,
            'applied_fixes': [],
            'manual_fixes_needed': []
        }

        if auto_fix:
            for fix in analysis.get('fixes', []):
                if self._is_safe_fix(fix):
                    success = self.code_fixer.apply_fix(analysis['file'], fix)
                    if success:
                        results['applied_fixes'].append(fix)
                    else:
                        results['manual_fixes_needed'].append(fix)
                else:
                    results['manual_fixes_needed'].append(fix)

        return results

    def debug_code(self, code: str, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Debug Python code by attempting to compile and run it.

        Args:
            code: Python code to debug
            auto_fix: Whether to apply automatic fixes

        Returns:
            Debugging results
        """
        results = {
            'syntax_errors': [],
            'runtime_errors': [],
            'applied_fixes': [],
            'manual_fixes_needed': []
        }

        # First check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            analysis = self.error_analyzer.analyze_error(e)
            results['syntax_errors'].append(analysis)

            if auto_fix:
                for fix in analysis.get('fixes', []):
                    if self._is_safe_fix(fix):
                        # For code strings, we can't apply fixes directly
                        results['manual_fixes_needed'].append(fix)

        # Try to execute the code
        try:
            exec(code)
        except Exception as e:
            analysis = self.error_analyzer.analyze_error(e)
            results['runtime_errors'].append(analysis)

            if auto_fix:
                for fix in analysis.get('fixes', []):
                    if self._is_safe_fix(fix):
                        results['manual_fixes_needed'].append(fix)  # Can't auto-fix runtime errors in strings

        return results

    def debug_file(self, file_path: str, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Debug a Python file.

        Args:
            file_path: Path to the Python file
            auto_fix: Whether to apply automatic fixes

        Returns:
            Debugging results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            results = self.debug_code(code, auto_fix=False)  # Don't auto-fix code strings

            # Apply fixes to the actual file
            if auto_fix:
                all_fixes = []
                for error_list in [results['syntax_errors'], results['runtime_errors']]:
                    for analysis in error_list:
                        all_fixes.extend(analysis.get('fixes', []))

                applied_fixes = []
                manual_fixes = []

                for fix in all_fixes:
                    if self._is_safe_fix(fix):
                        success = self.code_fixer.apply_fix(file_path, fix)
                        if success:
                            applied_fixes.append(fix)
                        else:
                            manual_fixes.append(fix)
                    else:
                        manual_fixes.append(fix)

                results['applied_fixes'] = applied_fixes
                results['manual_fixes_needed'] = manual_fixes

            return results

        except Exception as e:
            return {
                'error': f'Could not read file {file_path}: {str(e)}',
                'syntax_errors': [],
                'runtime_errors': [],
                'applied_fixes': [],
                'manual_fixes_needed': []
            }

    def _is_safe_fix(self, fix: Dict[str, Any]) -> bool:
        """Determine if a fix is safe to apply automatically."""
        safe_fix_types = ['add_import']  # Only allow safe fixes

        return fix.get('type') in safe_fix_types

    def get_error_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of debugging results."""
        summary_parts = []

        syntax_errors = results.get('syntax_errors', [])
        runtime_errors = results.get('runtime_errors', [])

        if syntax_errors:
            summary_parts.append(f"Found {len(syntax_errors)} syntax error(s)")

        if runtime_errors:
            summary_parts.append(f"Found {len(runtime_errors)} runtime error(s)")

        applied_fixes = results.get('applied_fixes', [])
        manual_fixes = results.get('manual_fixes_needed', [])

        if applied_fixes:
            summary_parts.append(f"Applied {len(applied_fixes)} automatic fix(es)")

        if manual_fixes:
            summary_parts.append(f"{len(manual_fixes)} fix(es) need manual attention")

        # Add suggestions
        all_suggestions = []
        for error_list in [syntax_errors, runtime_errors]:
            for analysis in error_list:
                all_suggestions.extend(analysis.get('suggestions', []))

        if all_suggestions:
            summary_parts.append("Suggestions:")
            for suggestion in all_suggestions[:5]:  # Limit to 5 suggestions
                summary_parts.append(f"- {suggestion}")

        return "\n".join(summary_parts) if summary_parts else "No issues found"