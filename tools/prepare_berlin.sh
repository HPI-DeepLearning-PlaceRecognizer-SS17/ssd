#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset berlin --set train --target $DIR/../data/train.lst --root $DIR/../data
python $DIR/prepare_dataset.py --dataset berlin --set val --target $DIR/../data/val.lst --root $DIR/../data --shuffle False