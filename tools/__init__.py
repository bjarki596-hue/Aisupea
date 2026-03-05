"""
Aisupea Tool System

Tools for the autonomous agent to interact with its environment.
"""

import os
import subprocess
import sys
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path


class Tool:
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError("Subclasses must implement execute method")

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter specifications for the tool."""
        return {}


class PythonExecutionTool(Tool):
    """Tool for executing Python code."""

    def __init__(self):
        super().__init__(
            name="python_executor",
            description="Execute Python code and return the result"
        )

    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution result
        """
        try:
            # Create a restricted environment for execution
            # This is a simplified version - real implementation would use more security measures
            result = eval(code, {"__builtins__": {}}, {})

            return {
                'success': True,
                'result': str(result),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'code': {
                'type': 'string',
                'description': 'Python code to execute',
                'required': True
            },
            'timeout': {
                'type': 'integer',
                'description': 'Execution timeout in seconds',
                'default': 30,
                'required': False
            }
        }


class BashExecutionTool(Tool):
    """Tool for executing bash commands."""

    def __init__(self):
        super().__init__(
            name="bash_executor",
            description="Execute bash commands in the system shell"
        )

    def execute(self, command: str, cwd: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute bash command.

        Args:
            command: Command to execute
            cwd: Working directory
            timeout: Execution timeout in seconds

        Returns:
            Execution result
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'command': {
                'type': 'string',
                'description': 'Bash command to execute',
                'required': True
            },
            'cwd': {
                'type': 'string',
                'description': 'Working directory for command execution',
                'required': False
            },
            'timeout': {
                'type': 'integer',
                'description': 'Execution timeout in seconds',
                'default': 30,
                'required': False
            }
        }


class FileSystemTool(Tool):
    """Tool for filesystem operations."""

    def __init__(self):
        super().__init__(
            name="filesystem",
            description="Perform filesystem operations like reading, writing, and listing files"
        )

    def execute(self, operation: str, path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute filesystem operation.

        Args:
            operation: Operation to perform ('read', 'write', 'list', 'delete', 'exists')
            path: File or directory path
            content: Content for write operations

        Returns:
            Operation result
        """
        try:
            path_obj = Path(path)

            if operation == 'read':
                if path_obj.is_file():
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {
                        'success': True,
                        'content': content,
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'content': None,
                        'error': f'Path is not a file: {path}'
                    }

            elif operation == 'write':
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content or '')
                return {
                    'success': True,
                    'error': None
                }

            elif operation == 'list':
                if path_obj.is_dir():
                    items = [str(item) for item in path_obj.iterdir()]
                    return {
                        'success': True,
                        'items': items,
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'items': None,
                        'error': f'Path is not a directory: {path}'
                    }

            elif operation == 'delete':
                if path_obj.exists():
                    if path_obj.is_file():
                        path_obj.unlink()
                    else:
                        import shutil
                        shutil.rmtree(path)
                    return {
                        'success': True,
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Path does not exist: {path}'
                    }

            elif operation == 'exists':
                return {
                    'success': True,
                    'exists': path_obj.exists(),
                    'error': None
                }

            else:
                return {
                    'success': False,
                    'error': f'Unknown operation: {operation}'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'operation': {
                'type': 'string',
                'description': 'Operation to perform',
                'enum': ['read', 'write', 'list', 'delete', 'exists'],
                'required': True
            },
            'path': {
                'type': 'string',
                'description': 'File or directory path',
                'required': True
            },
            'content': {
                'type': 'string',
                'description': 'Content for write operations',
                'required': False
            }
        }


class ProjectAnalyzerTool(Tool):
    """Tool for analyzing project structure and code."""

    def __init__(self):
        super().__init__(
            name="project_analyzer",
            description="Analyze project structure, dependencies, and code quality"
        )

    def execute(self, operation: str, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute project analysis operation.

        Args:
            operation: Operation to perform ('structure', 'dependencies', 'imports')
            path: Project root path

        Returns:
            Analysis result
        """
        project_path = Path(path or '.')

        try:
            if operation == 'structure':
                structure = self._analyze_structure(project_path)
                return {
                    'success': True,
                    'structure': structure,
                    'error': None
                }

            elif operation == 'dependencies':
                deps = self._analyze_dependencies(project_path)
                return {
                    'success': True,
                    'dependencies': deps,
                    'error': None
                }

            elif operation == 'imports':
                imports = self._analyze_imports(project_path)
                return {
                    'success': True,
                    'imports': imports,
                    'error': None
                }

            else:
                return {
                    'success': False,
                    'error': f'Unknown operation: {operation}'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_structure(self, path: Path) -> Dict[str, Any]:
        """Analyze project structure."""
        structure = {
            'files': [],
            'directories': [],
            'total_files': 0,
            'total_dirs': 0
        }

        for item in path.rglob('*'):
            if item.is_file():
                structure['files'].append(str(item.relative_to(path)))
                structure['total_files'] += 1
            elif item.is_dir():
                structure['directories'].append(str(item.relative_to(path)))
                structure['total_dirs'] += 1

        return structure

    def _analyze_dependencies(self, path: Path) -> Dict[str, Any]:
        """Analyze project dependencies."""
        deps = {
            'python_files': [],
            'imports': set(),
            'requirements': []
        }

        # Find Python files
        for py_file in path.rglob('*.py'):
            deps['python_files'].append(str(py_file.relative_to(path)))

            # Extract imports (simplified)
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import_lines = [line.strip() for line in content.split('\n')
                                  if line.strip().startswith(('import ', 'from '))]
                    deps['imports'].update(import_lines)
            except:
                pass

        # Check for requirements.txt
        req_file = path / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    deps['requirements'] = [line.strip() for line in f if line.strip()]
            except:
                pass

        deps['imports'] = list(deps['imports'])
        return deps

    def _analyze_imports(self, path: Path) -> Dict[str, Any]:
        """Analyze import patterns."""
        imports = {
            'modules': {},
            'external_deps': set(),
            'internal_deps': set()
        }

        # This is a simplified analysis
        # Real implementation would use AST parsing

        return imports

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'operation': {
                'type': 'string',
                'description': 'Analysis operation to perform',
                'enum': ['structure', 'dependencies', 'imports'],
                'required': True
            },
            'path': {
                'type': 'string',
                'description': 'Project root path',
                'default': '.',
                'required': False
            }
        }


class CodeSearchTool(Tool):
    """Tool for searching code in the project."""

    def __init__(self):
        super().__init__(
            name="code_search",
            description="Search for code patterns, functions, and text in the project"
        )

    def execute(self, query: str, path: Optional[str] = None,
               file_pattern: str = "*.py", case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Execute code search.

        Args:
            query: Search query
            path: Search path
            file_pattern: File pattern to search in
            case_sensitive: Whether search is case sensitive

        Returns:
            Search results
        """
        import glob

        search_path = Path(path or '.')
        results = []

        try:
            # Find matching files
            pattern = str(search_path / file_pattern)
            files = glob.glob(pattern, recursive=True)

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')

                        for line_num, line in enumerate(lines, 1):
                            search_line = line if case_sensitive else line.lower()
                            search_query = query if case_sensitive else query.lower()

                            if search_query in search_line:
                                results.append({
                                    'file': file_path,
                                    'line': line_num,
                                    'content': line.strip(),
                                    'context': self._get_context(lines, line_num - 1, 2)
                                })

                except Exception as e:
                    # Skip files that can't be read
                    continue

            return {
                'success': True,
                'results': results,
                'total_matches': len(results),
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'results': [],
                'total_matches': 0,
                'error': str(e)
            }

    def _get_context(self, lines: List[str], line_idx: int, context_lines: int) -> List[str]:
        """Get context lines around a match."""
        start = max(0, line_idx - context_lines)
        end = min(len(lines), line_idx + context_lines + 1)
        return [lines[i] for i in range(start, end)]

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'query': {
                'type': 'string',
                'description': 'Search query',
                'required': True
            },
            'path': {
                'type': 'string',
                'description': 'Search path',
                'default': '.',
                'required': False
            },
            'file_pattern': {
                'type': 'string',
                'description': 'File pattern to search in',
                'default': '*.py',
                'required': False
            },
            'case_sensitive': {
                'type': 'boolean',
                'description': 'Whether search is case sensitive',
                'default': False,
                'required': False
            }
        }


class CommandRouter(Tool):
    """Router tool that selects and executes other tools."""

    def __init__(self, tool_system: 'ToolSystem'):
        super().__init__(
            name="command_router",
            description="Route commands to appropriate tools and execute them"
        )
        self.tool_system = tool_system

    def execute(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Route and execute a command using appropriate tools.

        Args:
            command: Natural language command to execute
            **kwargs: Additional parameters

        Returns:
            Execution result
        """
        # Analyze command to determine which tool to use
        tool_name = self._analyze_command(command)

        if not tool_name:
            return {
                'success': False,
                'error': f'Could not determine appropriate tool for command: {command}'
            }

        # Get the tool
        tool = self.tool_system.get_tool(tool_name)
        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}'
            }

        # Extract parameters from command and kwargs
        params = self._extract_parameters(command, tool, kwargs)

        # Execute the tool
        return tool.execute(**params)

    def _analyze_command(self, command: str) -> Optional[str]:
        """Analyze command to determine which tool to use."""
        command_lower = command.lower()

        # Simple keyword-based routing
        if any(kw in command_lower for kw in ['run', 'execute', 'python', 'code']):
            return 'python_executor'
        elif any(kw in command_lower for kw in ['bash', 'shell', 'command', 'terminal']):
            return 'bash_executor'
        elif any(kw in command_lower for kw in ['file', 'read', 'write', 'list', 'delete']):
            return 'filesystem'
        elif any(kw in command_lower for kw in ['analyze', 'project', 'structure', 'dependencies']):
            return 'project_analyzer'
        elif any(kw in command_lower for kw in ['search', 'find', 'grep']):
            return 'code_search'

        return None

    def _extract_parameters(self, command: str, tool: Tool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for tool execution."""
        # This is a simplified parameter extraction
        # Real implementation would use NLP to parse parameters from command

        params = {}

        # For now, just pass through kwargs and try to extract from command
        if 'python_executor' in tool.name and 'code' not in kwargs:
            # Try to extract code from command
            if 'run' in command or 'execute' in command:
                # Very basic extraction
                code_start = command.find('```python')
                if code_start != -1:
                    code_end = command.find('```', code_start + 9)
                    if code_end != -1:
                        params['code'] = command[code_start + 9:code_end].strip()
                else:
                    # Assume the rest is code
                    params['code'] = command.split('run')[1].strip() if 'run' in command else command

        # Merge with provided kwargs
        params.update(kwargs)

        return params

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'command': {
                'type': 'string',
                'description': 'Natural language command to execute',
                'required': True
            }
        }


class ToolSystem:
    """Manages all available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

        # Register built-in tools
        self.register_tool(PythonExecutionTool())
        self.register_tool(BashExecutionTool())
        self.register_tool(FileSystemTool())
        self.register_tool(ProjectAnalyzerTool())
        self.register_tool(CodeSearchTool())

        # Command router (registered last so it can reference the tool system)
        router = CommandRouter(self)
        self.register_tool(router)

    def register_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}'
            }

        return tool.execute(**kwargs)

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            return None

        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.get_parameters()
        }