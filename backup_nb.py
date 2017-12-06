#!/usr/bin/env python3
#
# A hook script to create a backup copy of .ipynb files without deleting the
# original. The original might have output inside it, which is stripped out.

import os

files = os.listdir("src")

for fname in filter(lambda f: f.endswith(".ipynb") and "_backup" not in f, files):
    backup_fname = fname[:-6] + "_backup.ipynb"
    cmd = "cat src/{0} | nbstripout > src/{1}".format(fname, backup_fname)
    os.system(cmd)

os.system("git add --force src/*_backup.ipynb")
