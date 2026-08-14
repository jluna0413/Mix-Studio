import shutil
import os
import sys
import subprocess

src = r'D:\apps\Mix Studio\data'
dst = r'E:\MixStudioData'

print(f'Migrating data from {src} to {dst}...')

if not os.path.exists(src):
    print('Source directory not found:', src)
    sys.exit(1)

# Remove existing destination if it exists (to avoid copytree errors)
if os.path.exists(dst):
    try:
        subprocess.run(['rmdir', '/s', '/q', dst], check=False)
        print(f'Removed existing {dst} directory')
    except Exception as e:
        print(f'Warning: Could not remove {dst}: {e}')

try:
    shutil.copytree(src, dst)
    print('Migration complete!')
except Exception as e:
    print(f'Error during migration: {e}')
    sys.exit(1)
