import click
import inquirer
import sys
import os.path
from paperqa import Docs

# Contants
STORE_DIR = "store"
PDF_URL = "Fetch a PDF from a URL"
FILE = "Load from a file path (e.g. /path/to/file.pdf)"
ZOTERO_CLIPBOARD = "Read and parse PDF files from Zotero clipboard data (CTRL+SHIFT+C)" 

@click.command()
@click.argument('file_path', type=click.Path(exists=True), required=False, default=None, metavar='FILE_PATH')

def main(file_path):
    # Print the welcome message
    click.echo("Welcome to the PaperQA CLI!")
    click.echo("This is a tool for answering any question you may have about scientific papers!")

    # initialize the answer and file_path variables to None
    answer = None
    file_path = None

    # check if any command-line arguments were provided
    if sys.argv:
        # if exactly one argument was provided, assume it is the file path
        if len(sys.argv) == 2:
            arg_value = sys.argv[1]
            # check to make sure the file_path is valid
            if os.path.isfile(arg_value):
                answer = FILE
                file_path = arg_value
            else:
                click.echo("Warning: The file path you provided is not valid.")

    if answer is None:
        # ask the user how they want to load the PDF
        answer = inquirer.prompt(
            [inquirer.List("type", message="How can I help you today?", choices=[
                PDF_URL, 
                FILE, 
                ZOTERO_CLIPBOARD,
                # TODO: Add more options, such as ability to query a webpage from URL, a github repo, YouTube video, etc.
                ])])["type"]

    # handle user input
    if answer == PDF_URL:
        raise NotImplementedError("This feature is not yet implemented. Please try again later.")
    elif answer == FILE:
        # initialize the docs object
        docs = Docs()
        # check if the file path is valid
        if file_path is None:
            # request user input
            file_path = inquirer.prompt([inquirer.Path("path", message="Please enter a valid PDF file path (e.g. path/to/your.pdf).")])["path"]
        # add the PDF file to the docs
        docs.add(file_path)
    elif answer == ZOTERO_CLIPBOARD:
        raise NotImplementedError("This feature is not yet implemented. Please try again later.")
    else:
        raise ValueError("An unexpected error occurred.")
    
    # start the asking process
    docs.chat()

if __name__ == '__main__':
    main()


    
