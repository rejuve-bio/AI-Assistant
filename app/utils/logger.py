import logging
import time
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.traceback import install
import sys

# Install rich traceback handler
install(show_locals=False)

# Rich console instance
console = Console()

class RichLogger:
    """
    A custom logger that uses Rich to create beautiful, structured terminal output
    for LangGraph agent execution.
    """
    
    @staticmethod
    def setup_logging():
        pass

    @staticmethod
    def log_workflow_start(user_query: str):
        """Log the start of a workflow execution."""
        console.print()
        console.print(Panel(
            Text(user_query, justify="center", style="bold cyan"),
            title="[bold green]🚀 New Workflow Started",
            border_style="green",
            padding=(1, 2)
        ))
        console.print()

    @staticmethod
    def log_plan(plan: List[Any]):
        """Log the generated execution plan."""
        tree = Tree("📋 [bold yellow]Execution Plan[/bold yellow]")
        
        for i, step in enumerate(plan):
            agent = step.get("agent", "Unknown Agent").replace("_", " ").title()
            tree.add(f"[cyan]Step {i+1}:[/cyan] {agent}")
            
        console.print(tree)
        console.print()

    @staticmethod
    def log_group_start(group_idx: int, mode: str, agents: List[str]):
        """Log the start of an execution group (Parallel or Sequential)."""
        if mode == "parallel":
            header = f"🔀 Starting Group {group_idx + 1} (Parallel)"
            style = "bold magenta"
            tree = Tree(f"[{style}]{header}[/{style}]")
            for agent in agents:
                tree.add(f"[cyan]Running:[/cyan] {agent.replace('_', ' ').title()}")
            console.print(tree)
        else:
            header = f"➡️  Starting Group {group_idx + 1} (Sequential)"
            console.print(f"[bold blue]{header}[/bold blue] with agents: {', '.join(agents)}")

    @staticmethod
    def log_agent_start(agent_name: str, step_idx: int, total_steps: int):
        """Log when an agent starts running."""
        formatted_name = agent_name.replace("_", " ").title()
        console.print(f"[dim]Step {step_idx}/{total_steps}:[/dim] [bold cyan]Running {formatted_name}...[/bold cyan]")

    @staticmethod
    def log_agent_complete(agent_name: str, result: str, success: bool = True):
        """Log when an agent finishes."""
        formatted_name = agent_name.replace("_", " ").title()
        status_style = "bold green" if success else "bold red"
        status_icon = "✅" if success else "❌"
        status_text = "Completed" if success else "Failed"
        
        # Create a condensed output preview
        preview = result[:200] + "..." if len(result) > 200 else result
        preview = preview.replace("\n", " ")
        
        console.print(
            f"  {status_icon} [{status_style}]{formatted_name}[/{status_style}] {status_text}: "
            f"[dim]{preview}[/dim]"
        )

    @staticmethod
    def log_router_decision(agent_name: str, reason: str = ""):
        """Log a routing decision."""
        console.print(f"[yellow]⚡ Routing to:[/yellow] [bold]{agent_name}[/bold] {f'({reason})' if reason else ''}")

    @staticmethod
    def log_error(context: str, error_msg: str):
        """Log an error beautifully."""
        console.print(Panel(
            f"[bold red]Error in {context}:[/bold red]\n{error_msg}",
            title="❌ Error",
            border_style="red"
        ))

    @staticmethod
    def log_agent_called(message: str, user_id: str, content_ids: Any, graph_id: Any, urls: Any):
        """Highlight incoming agent requests."""
        console.print(
            "[bold magenta]Agent Called[/bold magenta]"
            f" [dim]user=[/dim]{user_id}"
            f" [dim]query=[/dim]{message}"
            f" [dim]content_ids=[/dim]{content_ids}"
            f" [dim]graph_id=[/dim]{graph_id}"
            f" [dim]urls=[/dim]{urls}"
        )

    @staticmethod
    def log_classifying_query(query: str):
        """Highlight classifier entry points."""
        console.print(f"[bold yellow]Classifying Query[/bold yellow] [dim]{query}[/dim]")

    @staticmethod
    def log_generated_plan(execution_groups: List[Dict[str, Any]]):
        """Highlight plan generation summary."""
        console.print(
            f"[bold green]Generated Plan[/bold green] "
            f"[dim]groups={len(execution_groups)}[/dim]"
        )

    @staticmethod
    def log_history_and_memory(history: List[Dict[str, Any]], memory: List[Any]):
        """Highlight loaded history/memory context."""
        console.print(
            f"[bold cyan]Conversation Context[/bold cyan] "
            f"[dim]history_items={len(history)} memory_items={len(memory)}[/dim]"
        )
