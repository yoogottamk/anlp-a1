#!/bin/bash

set -o errexit -o pipefail

function download {
    curl -fSL \
        https://web.iiit.ac.in/~yoogottam.khandelwal/$1 \
        -o $1
}

declare -A hashes
hashes[cbow.ckpt]=4d8fabfe86a7a75545e41c45a75446d1
hashes[com-feat.npy]=466522d580d45d8397e6392bb91c6804
hashes[w2i.pkl]=7e06cc25717ff5c1dc6637470b154ea8
hashes[wf.pkl]=223e078bec2613fb95738a2aa16b87cd

for file in cbow.ckpt com-feat.npy w2i.pkl wf.pkl; do
    [[ $( md5sum $file | cut -d' ' -f1 ) = ${hashes[$file]} ]] || download $file
done
