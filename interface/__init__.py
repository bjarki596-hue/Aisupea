"""
Aisupea Interface System

Interfaces for interacting with the AI system including CLI chat,
command interface, logging, and session management.
"""

import sys
import threading
import time
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import json


class Logger:
    """Logging system for the AI framework."""

    def __init__(self, log_file: Optional[str] = None, level: str = "INFO"):
        self.log_file = Path(log_file) if log_file else None
        self.level = level
        self.levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a message."""
        if self.levels.get(level, 0) < self.levels.get(self.level, 1):
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "extra": extra or {}
        }

        # Format for console
        console_msg = f"[{timestamp}] {level}: {message}"
        if extra:
            console_msg += f" {extra}"

        print(console_msg)

        # Write to file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.log("DEBUG", message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.log("INFO", message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.log("WARNING", message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.log("ERROR", message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        self.log("CRITICAL", message, extra)


class SessionManager:
    """Manages user sessions and conversation history."""

    def __init__(self, session_dir: str = ".aisu_sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.current_session: Optional[Dict[str, Any]] = None
        self.session_file: Optional[Path] = None

    def create_session(self, session_name: Optional[str] = None) -> str:
        """Create a new session."""
        if session_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"

        session_id = session_name
        self.session_file = self.session_dir / f"{session_id}.json"

        self.current_session = {
            "session_id": session_id,
            "created_at": time.time(),
            "messages": [],
            "metadata": {}
        }

        self._save_session()
        return session_id

    def load_session(self, session_id: str) -> bool:
        """Load an existing session."""
        session_file = self.session_dir / f"{session_id}.json"

        if not session_file.exists():
            return False

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                self.current_session = json.load(f)
                self.session_file = session_file
                return True
        except Exception:
            return False

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the current session."""
        if not self.current_session:
            self.create_session()

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        self.current_session["messages"].append(message)
        self._save_session()

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from current session."""
        if not self.current_session:
            return []

        messages = self.current_session["messages"]
        if limit:
            messages = messages[-limit:]

        return messages

    def get_recent_context(self, max_tokens: int = 1000) -> str:
        """Get recent conversation context."""
        messages = self.get_messages(limit=20)  # Last 20 messages

        context_parts = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_text = f"{msg['role']}: {msg['content']}\n"
            msg_tokens = len(msg_text.split())

            if current_tokens + msg_tokens > max_tokens:
                break

            context_parts.insert(0, msg_text)
            current_tokens += msg_tokens

        return "".join(context_parts).strip()

    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        session_files = self.session_dir.glob("*.json")
        return [f.stem for f in session_files]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_file = self.session_dir / f"{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            if self.current_session and self.current_session["session_id"] == session_id:
                self.current_session = None
                self.session_file = None
            return True

        return False

    def _save_session(self):
        """Save current session to file."""
        if self.current_session and self.session_file:
            try:
                with open(self.session_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_session, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Silently fail for now


class CLIChatInterface:
    """Command-line chat interface for interacting with the AI."""

    def __init__(self, inference_engine, session_manager: Optional[SessionManager] = None,
                 logger: Optional[Logger] = None):
        self.inference_engine = inference_engine
        self.session_manager = session_manager or SessionManager()
        self.logger = logger or Logger()

        self.running = False
        self.streaming = False

    def start_chat(self, session_name: Optional[str] = None):
        """Start an interactive chat session."""
        if session_name:
            if not self.session_manager.load_session(session_name):
                print(f"Creating new session: {session_name}")
                self.session_manager.create_session(session_name)
        else:
            session_id = self.session_manager.create_session()
            print(f"Started new session: {session_id}")

        self.running = True
        print("\n🤖 Aisupea AI Assistant")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)

        while self.running:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if self._handle_command(user_input):
                    continue

                # Add user message to session
                self.session_manager.add_message("user", user_input)

                # Get AI response
                print("\nAssistant: ", end="", flush=True)

                response = self._get_response(user_input)

                print(response)

                # Add assistant response to session
                self.session_manager.add_message("assistant", response)

                self.logger.info("Chat interaction", {
                    "user_input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    "response_length": len(response)
                })

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                self.logger.error(f"Chat error: {str(e)}")
                print(f"\nError: {str(e)}")

    def _handle_command(self, user_input: str) -> bool:
        """Handle special commands."""
        command = user_input.lower().strip()

        if command in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            self.running = False
            return True

        elif command in ['help', 'h', '?']:
            self._show_help()
            return True

        elif command.startswith('session'):
            parts = command.split()
            if len(parts) > 1:
                subcommand = parts[1]
                if subcommand == 'list':
                    sessions = self.session_manager.list_sessions()
                    if sessions:
                        print("Available sessions:")
                        for session in sessions:
                            print(f"  - {session}")
                    else:
                        print("No sessions found.")
                elif subcommand == 'load' and len(parts) > 2:
                    session_name = parts[2]
                    if self.session_manager.load_session(session_name):
                        print(f"Loaded session: {session_name}")
                    else:
                        print(f"Session not found: {session_name}")
                elif subcommand == 'new' and len(parts) > 2:
                    session_name = parts[2]
                    self.session_manager.create_session(session_name)
                    print(f"Created new session: {session_name}")
            return True

        elif command == 'clear':
            # Clear screen (Unix-like systems)
            print("\033[2J\033[H", end="")
            return True

        elif command == 'history':
            messages = self.session_manager.get_messages(limit=10)
            if messages:
                print("\nRecent conversation:")
                for msg in messages:
                    role = msg['role']
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"{role}: {content}")
            else:
                print("No conversation history.")
            return True

        return False

    def _show_help(self):
        """Show help information."""
        help_text = """
Available commands:
  help, h, ?     - Show this help
  quit, exit, q  - Exit the chat
  clear          - Clear the screen
  history        - Show recent conversation
  session list   - List all sessions
  session load <name>  - Load a session
  session new <name>   - Create a new session

Just type your message for regular chat.
        """
        print(help_text)

    def _get_response(self, user_input: str) -> str:
        """Get AI response for user input."""
        # Get conversation context
        context = self.session_manager.get_recent_context(max_tokens=500)

        # Build prompt with context
        if context:
            prompt = f"Context:\n{context}\n\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"

        # Generate response
        try:
            response = self.inference_engine.generate(prompt, max_length=200)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error generating a response."


class CommandInterface:
    """Command-line interface for executing specific commands."""

    def __init__(self, tool_system, logger: Optional[Logger] = None):
        self.tool_system = tool_system
        self.logger = logger or Logger()

    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Execute a command using the tool system."""
        try:
            result = self.tool_system.execute_tool("command_router", command=command, **kwargs)

            self.logger.info("Command executed", {
                "command": command,
                "success": result.get("success", False)
            })

            return result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e)
            }

            self.logger.error(f"Command execution error: {str(e)}")
            return error_result

    def run_interactive(self):
        """Run interactive command mode."""
        print("🤖 Aisupea Command Interface")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                command = input("\nCommand: ").strip()

                if not command:
                    continue

                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break

                if command.lower() in ['help', 'h', '?']:
                    self._show_command_help()
                    continue

                # Execute command
                result = self.execute_command(command)

                # Display result
                if result.get("success"):
                    print("✅ Success:")
                    if "result" in result:
                        print(result["result"])
                    elif "stdout" in result:
                        print(result["stdout"])
                    elif "items" in result:
                        for item in result["items"]:
                            print(f"  {item}")
                else:
                    print("❌ Error:")
                    print(result.get("error", "Unknown error"))

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")

    def _show_command_help(self):
        """Show command interface help."""
        help_text = """
Aisupea Command Interface

Execute natural language commands using available tools:

Examples:
  "run python code: print('hello world')"
  "list files in current directory"
  "search for 'class' in .py files"
  "analyze the project structure"
  "execute bash command: ls -la"

Available tools:
        """
        tools = self.tool_system.get_available_tools()
        for tool_name in tools:
            tool_info = self.tool_system.get_tool_info(tool_name)
            if tool_info:
                help_text += f"  - {tool_name}: {tool_info['description']}\n"

        help_text += "\nType 'quit' to exit."
        print(help_text)


class ProgressTracker:
    """Tracks progress of long-running operations."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

    def start_task(self, task_id: str, description: str, total_steps: Optional[int] = None):
        """Start tracking a task."""
        self.active_tasks[task_id] = {
            "description": description,
            "total_steps": total_steps,
            "current_step": 0,
            "start_time": time.time(),
            "status": "running"
        }

        self.logger.info(f"Started task: {description}", {"task_id": task_id})

    def update_task(self, task_id: str, step: Optional[int] = None, message: Optional[str] = None):
        """Update task progress."""
        if task_id not in self.active_tasks:
            return

        task = self.active_tasks[task_id]

        if step is not None:
            task["current_step"] = step

        if message:
            task["last_message"] = message

    def complete_task(self, task_id: str, result: Optional[Any] = None):
        """Mark task as completed."""
        if task_id not in self.active_tasks:
            return

        task = self.active_tasks[task_id]
        task["status"] = "completed"
        task["end_time"] = time.time()
        task["result"] = result

        duration = task["end_time"] - task["start_time"]
        self.logger.info(f"Completed task: {task['description']}", {
            "task_id": task_id,
            "duration": duration
        })

    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        if task_id not in self.active_tasks:
            return

        task = self.active_tasks[task_id]
        task["status"] = "failed"
        task["end_time"] = time.time()
        task["error"] = error

        duration = task["end_time"] - task["start_time"]
        self.logger.error(f"Failed task: {task['description']}", {
            "task_id": task_id,
            "duration": duration,
            "error": error
        })

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        return self.active_tasks.get(task_id)

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks."""
        return [task for task in self.active_tasks.values() if task["status"] == "running"]