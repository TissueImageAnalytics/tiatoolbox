"""Console script for tiatoolbox."""
from tiatoolbox.utils import misc_utils
import sys
import click


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """
    Computational pathology toolbox designed by TIALAB.
    """
    return 0


@main.command()
@click.option("--count", "-c", default=1, help="Number of times to print the input")
@click.option("--name", "-n", help="Print the output")
def hello(count, name):
    """
    prints the command "count" times.
    Args:
        count: Number of times to print the input
        name: Print the output

    Returns:
        Prints the input

    """
    misc_utils.hello(count=count, name=name)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
