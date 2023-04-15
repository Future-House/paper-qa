import inquirer
from paperqa import Docs
from paperqa.utils import start_asking
import sys
import os.path

# Contants
STORE_DIR = "store"
PDF_URL = "Fetch a PDF from a URL"
FILE = "Load from a file path (e.g. /path/to/file.pdf)"
ZOTERO_CLIPBOARD = "Read and parse PDF files from Zotero clipboard data (CTRL+SHIFT+C)" 

def main():
    # Print welcome message    
    print("")
    print("Welcome to paper-qa!")
    print("")

    answer = None
    file_path = None

    if sys.argv:
        if len(sys.argv) == 2:
            arg_value = sys.argv[1]
            if os.path.isfile(arg_value):
                answer = FILE
                file_path = arg_value

    if answer is None:
        # Ask if user wants to clone new repo or load existing index
        answer = inquirer.prompt(
            [inquirer.List("type", message="How can I help you today?", choices=[
                PDF_URL,
                FILE,
                ZOTERO_CLIPBOARD,
                # TODO: Add more options, such as ability to query a webpage from URL, a github repo, YouTube video, etc.
            ])]
        )["type"]

    # Handle user input
    if answer == PDF_URL:
        # initialize the docs object
        docs = Docs()

        # Ask for user input for URL
        url = input("Paste URL: ")

        # add the pdf to the docs object
        docs.add_pdf_from_url(url)    
    elif answer == FILE:
        # Ask for user input for file path
        if file_path is None:
            file_path = input("Paste file path: ")

        # add the file path to the docs object
        docs = Docs().add(file_path)
    elif answer == ZOTERO_CLIPBOARD:
        # initialize the docs object
        docs = Docs()

        # add the pdf data to the docs object
        docs.add_from_zotero_clipboard() # function will automatically paste data from zotero clipboard    
    else:
        print("Invalid input, exiting...")
        return

    # Start asking for queries
    start_asking(docs)

if __name__ == "__main__":
    main()
