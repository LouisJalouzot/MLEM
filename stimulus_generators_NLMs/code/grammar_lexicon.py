import lexicon_English
from utils import extract_verb


def get_grammar_lexicon(n_words_per_pos):
    Words = lexicon_English.get_Words(n_words_per_pos)

    grammar_lexicon = ""

    # Question Mark
    grammar_lexicon += "QM -> '?'\n"

    # Det and possessives
    grammar_lexicon += "Det -> 'the'\n"

    for role in ["subj", "obj"]:
        grammar_lexicon += f"""
    poss_{role}[NUM=sg, PERS=1, BOUND=TESTFEATUREMATCH] -> 'my'
    poss_{role}[NUM=pl, PERS=1, BOUND=TESTFEATUREMATCH] -> 'our'
    poss_{role}[PERS=2, BOUND=TESTFEATUREMATCH] -> 'your'
    poss_{role}[NUM=sg, GEN=f, PERS=3, BOUND=TESTFEATUREMATCH, ANIM=true] -> 'her'
    poss_{role}[NUM=sg, GEN=m, PERS=3, BOUND=TESTFEATUREMATCH, ANIM=true] -> 'his'
    poss_{role}[NUM=sg, GEN=m, PERS=3, BOUND=TESTFEATUREMATCH, ANIM=false] -> 'its'
    poss_{role}[NUM=pl, PERS=3, BOUND=TESTFEATUREMATCH] -> 'their'
    """

    # prepositions, complementizers and question words
    grammar_lexicon += f"""
    prep_P -> '{"'|'".join(Words['loc_preps'])}'
    subj_who[ANIM=true, CLAUSE=subjwho, PERS=3] -> 'who'
    embedsubj_who[ANIM=true, CLAUSE=subjwho, PERS=3] -> 'who'
    obj_who[ANIM=true, CLAUSE=objwho, PERS=3] -> 'who'
    embedobj_who[ANIM=true, CLAUSE=objwho, PERS=3] -> 'who'
    who[ANIM=true, PERS=3] -> 'who'
    subj_which[CLAUSE=subjwhich, PERS=3] -> 'which'
    embedsubj_which[CLAUSE=subjwhich, PERS=3] -> 'which'
    obj_which[CLAUSE=objwhich, PERS=3] -> 'which'
    embedobj_which[CLAUSE=objwhich, PERS=3] -> 'which'
    which[PERS=3] -> 'which'
    rel_That[CLAUSE=that] -> 'that'
    rel_Whether[CLAUSE=whether] -> 'whether'
    """

    # subject pronouns
    grammar_lexicon += """
    subj_PRO[NUM=sg, PERS=1, ANIM=true]->'I'
    subj_PRO[PERS=2, ANIM=true]->'you'
    subj_PRO[NUM=sg, PERS=3, GEN=m, ANIM=true]->'he'
    subj_PRO[NUM=sg, PERS=3, GEN=f, ANIM=true]->'she'
    subj_PRO[NUM=sg, PERS=3, ANIM=false]->'it'
    subj_PRO[NUM=pl, PERS=1, ANIM=true]->'we'
    subj_PRO[NUM=pl, PERS=3, ANIM=true]->'they'
    """

    # object pronouns
    grammar_lexicon += """
    obj_PRO[NUM=sg, PERS=1, ANIM=true, REFL=false]->'me'
    obj_PRO[PERS=2, ANIM=true, REFL=false]->'you'
    obj_PRO[NUM=sg, GEN=m, PERS=3, ANIM=true, REFL=false]->'him'
    obj_PRO[NUM=sg, GEN=f, PERS=3, ANIM=true, REFL=false]->'her'
    obj_PRO[NUM=sg, PERS=3, ANIM=false, REFL=false]->'it'
    obj_PRO[NUM=pl, PERS=1, ANIM=true, REFL=false]->'us'
    obj_PRO[NUM=pl, PERS=3, ANIM=true, REFL=false]->'them'
    """
    # reflexive pronouns
    grammar_lexicon += """
    obj_PRO[NUM=sg, PERS=1, ANIM=true, REFL=true]->'myself'
    obj_PRO[PERS=2, NUM=sg, ANIM=true, REFL=true]->'yourself'
    obj_PRO[PERS=2, NUM=pl, ANIM=true, REFL=true]->'yourselves'
    obj_PRO[NUM=sg, GEN=m, PERS=3, ANIM=true, REFL=true]->'himself'
    obj_PRO[NUM=sg, GEN=f, PERS=3, ANIM=true, REFL=true]->'herself'
    obj_PRO[NUM=sg, PERS=3, ANIM=false, REFL=true]->'itself'
    obj_PRO[NUM=pl, PERS=1, ANIM=true, REFL=true]->'ourselves'
    obj_PRO[NUM=pl, PERS=3, ANIM=true, REFL=true]->'themselves'

    """

    # quantifiers
    for role in ["subj", "obj"]:
        for NUM_code, NUM_value in zip(["sg", "pl"], ["singular", "plural"]):
            grammar_lexicon += f"""quantifier_{role}[NUM={NUM_code}] -> '{"'|'".join(Words['quantifiers'][NUM_value])}'\n"""
    grammar_lexicon += "\n"

    # nouns animate
    for role in ["subj", "obj", "embedsubj", "embedobj", "matrixsubj"]:
        for NUM_code, NUM_value in zip(["sg", "pl"], ["singular", "plural"]):
            for GEN_code, GEN_value in zip(["f", "m"], ["feminine", "masculine"]):
                grammar_lexicon += f"""
    {role}_N[NUM={NUM_code}, GEN={GEN_code}, PERS=3, ANIM=true] -> '{"'|'".join(Words['nouns'][GEN_value][NUM_value])}'"""
    grammar_lexicon += "\n"

    # proper names
    for role in ["subj", "obj", "embedsubj", "embedobj"]:
        NUM_code, NUM_value = "sg", "singular"
        for GEN_code, GEN_value in zip(["f", "m"], ["feminine", "masculine"]):
            grammar_lexicon += f"""
    {role}_PropN[NUM={NUM_code}, GEN={GEN_code}, PERS=3, ANIM=true]-> '{"'|'".join(Words['proper_names'][NUM_value][GEN_value])}'"""
    grammar_lexicon += "\n"

    # nouns inanimate
    for role in ["subj", "obj", "embedsubj", "embedobj"]:
        for NUM_code, NUM_value in zip(["sg", "pl"], ["singular", "plural"]):
            grammar_lexicon += f"""
    {role}_N[NUM={NUM_code}, PERS=3, ANIM=false] -> '{"'|'".join(Words['nouns_inanimate'][NUM_value])}'"""
    grammar_lexicon += "\n"

    # VERBS
    for role in ["verb", "embedverb"]:
        grammar_lexicon += extract_verb(
            f"{role}_Intrans", ", ANIM=true", "verbs_intran_anim"
        )
        grammar_lexicon += extract_verb(f"{role}_Intrans", "", "verbs_intran_inanim")
        grammar_lexicon += extract_verb(f"{role}_Trans", ", ANIM=true", "verbs")
    grammar_lexicon += extract_verb("matrixverb_V", ", ANIM=true", "matrix_verbs")
    grammar_lexicon += extract_verb("do_Aux", "", "do_Aux")

    # COPULA
    grammar_lexicon += """
    copula[NUM=sg] -> 'is'
    copula[NUM=pl] -> 'are'
    """

    return grammar_lexicon


if __name__ == "__main__":
    pass
