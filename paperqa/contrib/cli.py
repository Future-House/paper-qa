import click
import inquirer
import sys
import os.path
import glob
from paperqa import Docs

# Contants
STORE_DIR = "store"
PDF_URL = "Fetch a PDF from a URL"
FILES = "Load from file path(s) (e.g. /path/to/file.pdf or /path/to/*.pdf)"
ZOTERO_CLIPBOARD = "Read and parse PDF files from Zotero clipboard data (CTRL+SHIFT+C)" 

@click.command()
@click.argument('file_paths', nargs=-1, default=None, metavar='FILE_PATHS')

def main(file_paths):
    # Print the welcome message
    click.echo("Welcome to the PaperQA CLI!")
    click.echo("This is a tool for answering any question you may have about scientific papers!")

    # initialize the answer and file_path variables to None
    answer = None
    file_paths = []

    # check if any command-line arguments were provided
    if sys.argv:
        # if one ore more arguments is provided, assume they are file paths
        if len(sys.argv) >= 2:
            arg_values = sys.argv[1:]
            glob_args = glob.glob(arg_values[0], recursive=True)
            # check to make sure the file_paths are valid
            for arg in glob_args:
                if os.path.isfile(arg):
                    file_paths.append(arg)
                    answer = FILES
                else:
                    click.echo("Warning: The file path you provided is not valid.")
                    answer = None

    if answer is None:
        # ask the user how they want to load the PDF
        answer = inquirer.prompt(
            [inquirer.List("type", message="How can I help you today?", choices=[
                PDF_URL, 
                FILES, 
                ZOTERO_CLIPBOARD,
                # TODO: Add more options, such as ability to query a webpage from URL, a github repo, YouTube video, etc.
                ])])["type"]

    # handle user input
    if answer == PDF_URL:
        raise NotImplementedError("This feature is not yet implemented. Please try again later.")
    elif answer == FILES:
        # initialize the docs object
        docs = Docs()
        if len(file_paths) == 0:
            # request user input
            file_path_pattern = inquirer.prompt([inquirer.Path("path", message="Please enter a valid PDF file path or pattern (e.g. path/to/your.pdf or my_pdfs/**/*.pdf).")])["path"]
            file_paths = glob.glob(file_path_pattern, recursive=True)
        # add the PDF files to the docs
        for file_path in file_paths:
            docs.add(file_path)
    elif answer == ZOTERO_CLIPBOARD:
        raise NotImplementedError("This feature is not yet implemented. Please try again later.")
    else:
        raise ValueError("An unexpected error occurred.")
    
    # start the chatting with the documents!
    docs.chat()

if __name__ == '__main__':
    main()


    
