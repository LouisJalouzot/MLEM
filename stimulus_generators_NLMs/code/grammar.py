### On peut modifier le lexicon comme la grammaire

import argparse
import os
import grammar_lexicon

parser = argparse.ArgumentParser()

parser.add_argument(
    "--grammar",
    type=str,
    default="clause_type",
    help="Name of the file containing the grammar rules.",
)
parser.add_argument(
    "--lexicon",
    type=str,
    default="grammar_lexicon",
    help="Name of the file containing the grammar lexicon.",
)
parser.add_argument(
    "--nwords",
    type=int,
    default=10,
    help="Number of items per POS in the lexicon.",
)
args = parser.parse_args()

fn_grammar_rules = args.grammar + ".txt"
fn_grammar = "grammars/" + args.grammar + ".fcfg"
print("New file created:", fn_grammar)

grammar = ""
grammar += "####################\n# PRODUCTION RULES #\n####################\n\n"
grammar += open(os.path.join("grammar_rules", fn_grammar_rules), "r").read()
grammar += "\n#################\n# LEXICAL RULES #\n#################\n\n"
grammar += grammar_lexicon.get_grammar_lexicon(args.nwords)

with open(fn_grammar, "w+") as f:
    f.write(grammar)
