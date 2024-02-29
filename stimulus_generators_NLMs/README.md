# stimulus_generators_NLMs

## Setup

```bash
pip install wordfreq nltk plotnine
```

## Operations

- Change the rules of the grammar in grammar_rules/```dataset```.txt
- Change the lexical rules in grammar_lexicon.py
- Change the lexical items in lexicon_English.py
- Run ```bash stimuli_csv_generator.sh '```dataset```' 'grammar_lexicon' 10```

These operations will modify the following folders:
- grammar_rules/ the rules to create the sentences
- grammars/ the corresponding fcfg file
- stimuli_unfiltered/ the corresponding csv file before cleaning up
- stimuli/ the final csv file after cleaning up

Currently available:
- ```dataset```=short_sentence
- ```dataset```=relative_clause
- ```dataset```=long_range_agreement
- ```dataset```=clause_type (unused)

To get all of these, run:
```console
bash stimuli_csv_generator.sh 'short_sentence' 'grammar_lexicon' 10
bash stimuli_csv_generator.sh 'relative_clause' 'grammar_lexicon' 10
bash stimuli_csv_generator.sh 'long_range_agreement' 'grammar_lexicon' 10
bash stimuli_csv_generator.sh 'clause_type' 'grammar_lexicon' 2
```

## More flexibility

1. Run code/grammar.py to get => code/grammars/grammar.fcfg
> Argparser arguments:\
> --grammar grammar_rules : grammar rules from the grammar_rules folder \
> --lexicon grammar_lexicon : lexicon without the .py extension \
> --nwords 3 : max number of words per POS
2. Run code/stimulus_generator_from_CFG.py to get the sentences => stimuli_unfiltered/grammar.csv
> Argparser arguments:\
> --grammar grammar : fcfg file generated at step 1 in folder grammars.
3. Run code/stimulus_cleanup.py to clean the stimuli and add columns => stimuli/grammar.csv
> Argparser arguments:\
> --csv stimuli_from_fcfg : csv file generated at step 2 in folder stimuli_unfiltered.
4. To check if stimuli are correctly balanced use stimuli/check_balance.py:
> python3 stimuli/check_balance.py --csv stimuli/short_sentence.csv \
> --csv : csv file to check \
Optionally you may append a sublist of features, as in: \
> python3 check_balance.py --csv clause_type.csv -f sentence_CLAUSE rel \
To check everything:
python3 stimuli/check_balance.py --csv stimuli/short_sentence.csv
python3 stimuli/check_balance.py --csv stimuli/clause_type.csv
python3 stimuli/check_balance.py --csv stimuli/relative_clause.csv