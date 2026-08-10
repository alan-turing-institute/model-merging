#!/bin/bash
set -euo pipefail
SRC="$1"
DST="$2"

echo "=== Copying $SRC -> $DST ==="
cp -r "$SRC"/. "$DST"/

python3 -c "
import json
p = '$DST/config.json'
d = json.load(open(p))
print('before:', d.get('vocab_size'))
d['vocab_size'] = 32000
json.dump(d, open(p, 'w'), indent=2)
print('after:', d['vocab_size'])
"
echo "PATCH_COMPLETE"
