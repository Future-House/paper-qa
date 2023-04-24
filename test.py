from dotenv import load_dotenv
load_dotenv()

import code
import os
from paperqa.docs import Docs
import re
import asyncio

d = Docs()

IGNORE_DIRS = set(['venv', 'node_modules', 'data'])
RX_IGNORE = re.compile(r'(^\..*)|(.*\.pyc$)')
for code_dir in ['.']:
    for dirpath, dirnames, filenames in os.walk(code_dir, topdown=True):
        dirnames[:] = [x for x in dirnames if x not in IGNORE_DIRS and not x.startswith('.')]
        for file in filenames:
            if RX_IGNORE.match(file):
                continue
            path = os.path.join(dirpath, file)
            print("Adding", path)
            d.add(path, citation=path, key=path)

a = asyncio.get_event_loop().run_until_complete(
    d.aquery(
        "What does this codebase do?",
    )
)

print(len(a.contexts), 'contexts')
print('Cost:', a.cost, 'Tokens:', a.tokens)
print(a.formatted_answer)
print(a.context)

code.interact(local=locals(), banner="""
Welcome to the PaperQA REPL! Try this:
  a = d.query("What manufacturing challenges are unique to bispecific antibodies?")
  print(a.formatted_answer)
""")
