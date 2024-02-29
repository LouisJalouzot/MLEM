#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import utils
from utils import sanity_checks
from nltk.parse import load_parser
from nltk.parse.generate import generate
from tqdm import tqdm
import pandas as pd
import argparse

import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument(
    "--grammar",
    type=str,
    default="clause_type",
    help="Name of the .fcfg file containing the grammar.",
)
args = parser.parse_args()

path2grammar = os.path.join("grammars", args.grammar + ".fcfg")
outfile = os.path.join("../stimuli_unfiltered", args.grammar + ".csv")

fcfg = load_parser(path2grammar, trace=0)


def process_sentence(s):
    sentence = " ".join(s)
    for tree in fcfg.parse(s):  # enter loop only if parsable
        d = {}
        if sentence.split()[-1] != "?":
            sentence += " ."
        d["sentence"] = sentence
        sanity_checks(
            sentence, tree
        )  # e.g., agreement ('dog see') - not a full proof test

        # extract sentence features from tree label
        for i_item, (feature, val) in enumerate(tree.label().items()):
            # if feature == 'GROUP':
            d[f"sentence_{feature}"] = val  # add sentential features

        # extract word features from tree pos
        for pos in tree.pos():  # Loop over part of speech
            d = utils.add_features_to_dict(d, pos)

        return d
    return None


if __name__ == "__main__":
    if args.verbose:
        print(fcfg.grammar())

    utils.print_time()
    print("Generating all sentences...")
    sentences = list(generate(fcfg.grammar()))  # Exhausting generator for tqdm counter.
    print(f"- Number of sentences: {len(sentences)}")

    df = utils.sentences_to_df(sentences)

    utils.print_time("Removing duplicate sentences...")
    df = utils.remove_repeated_sentences(df)
    print(f"- Number of sentences: {len(df)}")

    utils.print_time("Removing clearly faulty agreements...")
    df = utils.remove_faulty_agreements(df)
    print(f"- Number of sentences: {len(df)}")

    utils.print_time("Removing sentences with duplicate lemmas...")
    df = utils.remove_sentences_with_repeated_lemma(df)
    print(f"- Number of sentences: {len(df)}")
    print(df)
    utils.print_time("Parsing sentences...")
    sentences = utils.df_to_sentences(df)
    process_pool = multiprocessing.Pool(processes=os.cpu_count())
    d_grammar = list(
        tqdm(process_pool.imap(process_sentence, sentences), total=len(sentences))
    )

    d_grammar = [s for s in d_grammar if s is not None]

    # To dataframe
    df = pd.DataFrame(d_grammar)

    df.to_csv(outfile)
    print(df)
    print(f"Stimuli saved to {outfile}")
