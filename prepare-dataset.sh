#!/bin/bash

set -o errexit -o pipefail

RAW_DATASET_HASH=e4524af6c644cd044b1969bac7b62b2a
EXTRACTED_DATASET_HASH=fc436f9e2bf6d1ac4f04ddd8f45e92da
CSV_DATASET_HASH=12945c8f6e03bc4f3c03f290555ce7f0

function download_dataset {
    curl -fSL \
        http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz \
        -oraw-dataset.json.gz
}

function extract_dataset {
    gunzip \
        < raw-dataset.json.gz \
        | jq --indent 0 '{review: .reviewText | ascii_downcase | gsub("[,.!?]"; " ") | gsub("[^ a-z]"; "")}' \
        | pv -F "[JSON] Rate: %r | Avg: %a | %t" \
        > ./dataset.json
}

function generate_csv_dataset {
    (
        head -1 dataset.json \
            | jq -r 'keys | @csv' && \
        jq -r '[.[]] | @csv' < dataset.json \
    ) | pv -F "[CSV] Rate: %r | Avg: %a | %t" \
        > dataset.csv
}

function generate_sqlite_dataset {
    sqlite3 dataset.sqlite <<EOF
.mode csv
.import dataset.csv dataset
EOF
}

[[ $( md5sum raw-dataset.json.gz | cut -d' ' -f1 ) = $RAW_DATASET_HASH ]] || download_dataset
[[ $( md5sum dataset.json | cut -d' ' -f1 ) = $EXTRACTED_DATASET_HASH ]] || extract_dataset
[[ $( md5sum dataset.csv | cut -d' ' -f1 ) = $CSV_DATASET_HASH ]] || generate_csv_dataset

generate_sqlite_dataset
