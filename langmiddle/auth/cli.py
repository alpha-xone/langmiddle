"""CLI commands for LangMiddle authentication."""

import click

from .session import (
    get_credentials_path,
    get_current_user,
    is_authenticated,
    login_user,
    logout_user,
    register_user,
)


@click.group()
def auth() -> None:
    """Manage LangMiddle authentication."""
    pass


@auth.command()
@click.option("--email", prompt=True, help="Your email address")
@click.option(
    "--password",
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
    help="Your password",
)
def register(email: str, password: str) -> None:
    """Register a new account on LangMiddle shared backend."""
    try:
        result = register_user(email, password, save_credentials=True)
        click.echo("✅ Account created successfully!")
        click.echo(f"   User ID: {result['user_id']}")
        click.echo(f"   Email: {result['email']}")
        click.echo(f"   Credentials saved to: {get_credentials_path()}")
    except Exception as e:
        click.echo(f"❌ Registration failed: {e}", err=True)
        raise click.Abort()


@auth.command()
@click.option("--email", prompt=True, help="Your email address")
@click.option("--password", prompt=True, hide_input=True, help="Your password")
def login(email: str, password: str) -> None:
    """Login to your LangMiddle account."""
    try:
        result = login_user(email, password, save_credentials=True)
        click.echo("✅ Logged in successfully!")
        click.echo(f"   User ID: {result['user_id']}")
        click.echo(f"   Email: {result['email']}")
    except Exception as e:
        click.echo(f"❌ Login failed: {e}", err=True)
        raise click.Abort()


@auth.command()
def logout() -> None:
    """Logout and clear saved credentials."""
    try:
        logout_user()
        click.echo("✅ Logged out successfully!")
    except Exception as e:
        click.echo(f"❌ Logout failed: {e}", err=True)


@auth.command()
def status() -> None:
    """Check authentication status."""
    if is_authenticated():
        user = get_current_user()
        if user:
            click.echo("✅ Authenticated")
            click.echo(f"   User ID: {user['user_id']}")
            click.echo(f"   Email: {user.get('email', 'N/A')}")
            click.echo(f"   Credentials: {get_credentials_path()}")
        else:
            click.echo("⚠️  Authentication status unclear")
    else:
        click.echo("❌ Not authenticated")
        click.echo("   Run: langmiddle auth register")


@auth.command()
def whoami() -> None:
    """Show current user information."""
    try:
        user = get_current_user()
        if user:
            click.echo(f"User ID: {user['user_id']}")
            click.echo(f"Email: {user.get('email', 'N/A')}")
        else:
            click.echo("Not logged in", err=True)
            raise click.Abort()
    except Exception:
        click.echo("Not logged in", err=True)
        raise click.Abort()
