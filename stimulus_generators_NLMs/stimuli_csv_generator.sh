#!/bin/bash

echo "Grammar file : grammar_rules/$1"
echo "Lexicon file : $2"
echo "n_words : $3"

cd code
python grammar.py --grammar $1 --lexicon $2 --nwords $3 #-> code/grammars/$1.fcfg
python stimulus_generator_from_CFG.py --grammar $1 #-> stimuli_unfiltered/$1.csv
python stimulus_cleanup.py --grammar $1 #-> stimuli/$1.csv

echo "Output stimuli file : stimuli/$1"

python ../stimuli/check_balance.py --csv ../stimuli/$1.csv