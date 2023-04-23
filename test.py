from dotenv import load_dotenv
load_dotenv()

import code
import os
from paperqa import Docs

docs = Docs()
import re

IGNORE_DIRS = set(['venv', 'node_modules'])
RX_IGNORE = re.compile(r'(^\..*)|(.*\.pyc$)')
for code_dir in ['.']:
    for dirpath, dirnames, filenames in os.walk(code_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith('.')]
        for file in filenames:
            if RX_IGNORE.match(file):
                continue
            path = os.path.join(dirpath, file)
            print("Adding", path)
            docs.add(path, citation=path, key=path)


code.interact(local=locals(), banner="""
Welcome to the PaperQA REPL! Try this:
  answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
  print(answer.formatted_answer)
""")
