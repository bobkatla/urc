import click
from urc import __version__
from urc.cli.quick_sanity_run import sanity

@click.group()
@click.version_option(version=__version__, prog_name="urc")
def main():
    """URC - Unified Region Calibration"""
    pass


@main.command()
def info():
    """Display information about the URC package."""
    click.echo(f"URC version {__version__}")
    click.echo("Unified Region Calibration framework.")


main.add_command(sanity, name="sanity")