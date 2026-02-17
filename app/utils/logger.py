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
        """Configure standard logging to use RichHandler if desired, or keep it separate."""
        # For now, we will use this class for specific execution events 
        # while keeping the standard logger for debug/info messsages.
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
