"""LangMiddle CLI - Command-line interface."""

import click

from .auth.cli import auth


@click.group()
@click.version_option()
def cli() -> None:
    """LangMiddle CLI - Memory middleware for LangChain/LangGraph."""
    pass


# Register auth commands
cli.add_command(auth)


if __name__ == "__main__":
    cli()
