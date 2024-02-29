#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--grammar",
    type=str,
    help="Name of the .csv file containing the stimuli. In the stimuli_unfiltered folder.",
)
args = parser.parse_args()

csvfile = "../stimuli_unfiltered/" + args.grammar + ".csv"
outfile = "../stimuli/" + args.grammar + ".csv"

df = pd.read_csv(csvfile, index_col=[0])
utils.print_time(f"> Loading dataframe: {csvfile}")

# temporary fix, now fixed in grammar.py
df = utils.fix_poss_anim(df)

utils.print_time("> Modifying columns...")
df = utils.modify_tense_of_obj_quest(df)

utils.print_time(f"> Adding new columns...")
utils.print_time("Clause type")
df = utils.add_clause_type(df)
utils.print_time("Word Zipf")
df = utils.add_word_zipf(df)
utils.print_time("Incongruence counts")
df = utils.calc_incongruence_counts(df)
utils.print_time("Sentence length")
df = utils.add_sentence_length(df)
utils.print_time("has_property")
df = utils.add_has_property(df)
utils.print_time("LR agreements")
df = utils.lr_agreement_with_attractor(df)
utils.print_time("Binding")
df = utils.add_binding(df)
utils.print_time("> Remove impossible binding")
df = utils.remove_impossible_binding(df)
utils.print_time("> Add lemmas and Zipf")
df = utils.add_lemmas_and_zipf(df)

utils.print_time("Re-arange columns")
df = utils.order_columns(df, ["sentence_length", "sentence_GROUP", "sentence"])

# Print and save
print(df)
df.to_csv(outfile)
utils.print_time(f"> Dataframe saved to: {outfile}")
