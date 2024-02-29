#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import re
import warnings

import numpy as np
import pandas as pd
import wordfreq
from lexicon_English import Words

warnings.filterwarnings(
    "ignore",
    "This pattern is interpreted as a regular expression, and has match groups.",
)


def print_time(message=None):
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S")
    if message is None:
        print(time_str, end="\t")
    else:
        print(f"{time_str}\t{message}")


def sentences_to_df(sentences):
    sentences_txt = [" ".join(sentence) for sentence in sentences]
    df = pd.DataFrame({"sentence": sentences_txt})
    return df


def df_to_sentences(df):
    sentences_txt = df["sentence"]
    sentences = [sentence_txt.split(" ") for sentence_txt in sentences_txt]
    return sentences


def godown_dict_keys(d: dict, ks):
    res = set()
    if type(d) is dict:
        for k, v in d.items():
            res = res.union(
                godown_dict_keys(v, [key for key in ks if (re.search(key, k) is None)])
            )
    else:
        if len(ks) == 0:
            res = res.union(set(d))
    return res


def get_leaves(d: dict):
    res = []
    if type(d) is dict:
        for k, v in d.items():
            res += get_leaves(v)
    else:
        res.append(d)
    return res


def get_all_lemmas(Words):
    pos_with_lemmas = [
        ["nouns", "masculine"],
        ["nouns", "feminine"],
        ["nouns_inanimate"],
        ["verbs"],
        ["verbs_intran_anim"],
        ["verbs_intran_inanim"],
        # ["copula"],
        # ["do_Aux"],
        ["matrix_verbs"],
        ["loc_preps"],
        # ["quantifiers"],
        # ["proper_names"],
    ]
    res = []
    for pos in pos_with_lemmas:
        words = Words
        for k in pos:
            words = words[k]
        variations = get_leaves(words)
        lemma_list = [list(x) for x in zip(*variations)]
        res += lemma_list
    return res


def calc_lemma(word, LEMMAS):
    for lemmas in LEMMAS:
        if word in lemmas:
            return lemmas[0]
    return word


def add_lemmas_and_zipf(df):
    LEMMAS = get_all_lemmas(Words)
    columns = [
        col
        for col in df.columns
        if (
            col.endswith("subj")
            | col.endswith("verb")
            | col.endswith("obj")
            | col.endswith("prep")
        )
    ]
    for col in columns:
        df[f"{col}_LEMMA"] = df[col].apply(calc_lemma, LEMMAS=LEMMAS)
        df[f"{col}_ZIPF"] = df[f"{col}_LEMMA"].apply(calc_zipf)
    return df


def calc_zipf(word):
    if pd.notnull(word):
        return wordfreq.zipf_frequency(word, lang="en")
    return np.nan


def reg_unigrams(w_list, border=r"\b"):
    return "|".join(rf"{border}{w}{border}" for w in w_list)


def reg_bigrams(w1_list, w2_list):
    return f"({reg_unigrams(w1_list)})" + r"\s" + f"({reg_unigrams(w2_list)})"


def remove_faulty_agreements(df):
    # Warning: adjency between noun and verb does not guarantee agreement
    # Here it is protected because we look at (noun, verb) pairs very early
    # Beware of RC, auxiliaries, etc.

    patterns_a = []

    pro_inanimate = ["it"]
    noun_inanimate = godown_dict_keys(Words, ["nouns_inanimate"])
    verb_animate = godown_dict_keys(
        Words, [r"\bverbs\b|\bverbs_intran_anim\b|\bmatrix_verbs\b"]
    )
    pattern_animacy = "[A-Za-z]+\s" + reg_bigrams(noun_inanimate, verb_animate)
    pattern_it_animacy = reg_bigrams(pro_inanimate, verb_animate)

    patterns_a.append(pattern_it_animacy)
    patterns_a.append(pattern_animacy)

    pro_verb_s = ["he", "she", "it"]
    noun_sg_anim = godown_dict_keys(
        Words, [r"(\bnouns\b|\bnouns_inanimate\b)", "singular"]
    )
    noun_sg_inanim = godown_dict_keys(Words, [r"(\bnouns_inanimate\b)", "singular"])
    noun_sg = noun_sg_anim.union(noun_sg_inanim)
    proper_names = godown_dict_keys(Words, [r"\bproper_names\b"])

    pro_verb_no_s = ["I", "you", "we", "they"]
    noun_pl = godown_dict_keys(Words, [r"\bnouns\b|\bnouns_inanimate\b", "plural"])

    verb_sg = godown_dict_keys(Words, ["verbs", "present", "singular"])
    verb_pl = godown_dict_keys(Words, ["verbs", "present", "plural"])

    pattern_PN_sg = reg_bigrams(proper_names, verb_pl)
    pattern_noun_sg = "[A-Za-z]+\s" + reg_bigrams(noun_sg, verb_pl)
    pattern_pro_sg = reg_bigrams(pro_verb_s, verb_pl)
    pattern_pl = "[A-Za-z]+\s" + reg_bigrams(noun_pl, verb_sg)
    pattern_pro_pl = reg_bigrams(pro_verb_no_s, verb_sg)

    patterns_a.append(pattern_PN_sg)
    patterns_a.append(pattern_noun_sg)
    patterns_a.append(pattern_pro_sg)
    patterns_a.append(pattern_pl)
    patterns_a.append(pattern_pro_pl)

    patterns = []
    for pattern in patterns_a:
        patterns.append(f"^{pattern}")
        patterns.append(f"(that|whether)\s{pattern}")
    patterns.append("which\s.*" + reg_bigrams(noun_sg, verb_pl))
    patterns.append("which\s" + reg_bigrams(noun_pl, verb_sg))
    patterns.append("which\s.*" + reg_bigrams(noun_inanimate, verb_animate))

    quant_sg = ["every", "no"]
    quant_pl = ["all", "few"]
    pattern_q_sg = reg_bigrams(quant_sg, noun_pl)
    pattern_q_pl = reg_bigrams(quant_pl, noun_sg)
    patterns.append(f"{pattern_q_sg}")
    patterns.append(f"{pattern_q_pl}")

    patterns.sort(key=len)

    for pattern in patterns:
        mask = df["sentence"].str.contains(pattern)
        df = df[~mask]

    return df


def extract_verb(POS, anim_feature, Wkey):
    if POS == "embedverb_Matrix":
        return ""
    W = Words[Wkey]
    res = ""
    res = f"""
{POS}[finite=true, TENSE=pres, NUM=sg, PERS=1{anim_feature}] -> '{"'|'".join(W['present']['plural'])}'
{POS}[finite=true, TENSE=pres, NUM=sg, PERS=2{anim_feature}] -> '{"'|'".join(W['present']['plural'])}'
{POS}[finite=true, TENSE=pres, NUM=sg, PERS=3{anim_feature}] -> '{"'|'".join(W['present']['singular'])}'
{POS}[finite=true, TENSE=pres, NUM=pl{anim_feature}] -> '{"'|'".join(W['present']['plural'])}'
"""
    # block future for now:
    # {POS}[finite=true, TENSE=future{anim_feature}] -> '{"'|'".join(W['future'])}'"""
    if "-finite" in W.keys():
        res += f"""
{POS}[finite=false{anim_feature}] -> '{"'|'".join(W['-finite'])}'\n"""
        res += "\n"
    if "past" in W.keys():
        res += f"""
{POS}[finite=true, TENSE=past{anim_feature}] -> '{"'|'".join(W['past'])}'\n"""
        res += "\n"
    return res


def fix_poss_anim(df):
    if "poss" in df:
        df.loc[df.poss == "her", "poss_ANIM"] = True
        df.loc[df.poss == "his", "poss_ANIM"] = True
    return df


def add_features_to_dict(d, pos_tuple):
    word, pos = pos_tuple
    for i_item, (key, val) in enumerate(pos.items()):
        if i_item == 0:  # Get subj/verb/object from *type* of pos
            svo = None
            if "_" in val:
                svo, svo_type = val.split("_")
                if svo_type == "N" and "embed" not in svo:
                    d[f"main_{svo}"] = word
                if svo_type != "":
                    d[f"{svo}_type"] = svo_type
                    d[f"{svo}"] = word
        else:  # Features of pos
            if svo is not None:
                feature_name = key
                d[f"{svo}_{feature_name}"] = val
    return d


def check_incongruence(f1, f2, role1=None, role2=None, poss_type=None):
    if pd.isnull(f1) or pd.isnull(f2):
        return np.nan
    elif role1 == "poss" and poss_type != role2:
        # Verify that the possessive is of the current role.
        # Since, poss-subj congruence means 'poss' is the possessive of the subject
        # And poss-obj congruence means the 'poss' is of the object.
        return np.nan
    else:
        return f1 != f2


def get_agreement_mismatch(row, role1, role2, agr_features=None):
    if agr_features is None:
        agr_features = ["GEN", "NUM", "PERS", "ANIM"]
    agreement_mismatch = {}
    if pd.isnull(row[f"{role1}_type"]) or pd.isnull(row[f"{role2}_type"]):
        # How do we mark that?
        # return np.nan?
        # agreement_match = {feature: np.nan for feature in agr_features}
        pass
    else:
        for feature in agr_features:
            agreement_mismatch[feature] = check_incongruence(
                row[f"{role1}_{feature}"], row[f"{role2}_{feature}"]
            )
    agreement_mismatch["overall"] = any(
        agr_value
        for agr_value in agreement_mismatch.values()
        if agr_value is not np.nan
    )
    return agreement_mismatch


def order_columns(df, bring2front):
    # sort columns by name
    cols = sorted(list(df))
    # Move sentence to front column and save
    for name in bring2front:
        if name in cols:
            cols.insert(0, cols.pop(cols.index(name)))
            df = df.loc[:, cols]
    return df


def remove_repeated_sentences(df):
    df = df.drop_duplicates(subset=["sentence"])
    return df


def remove_sentences_with_repeated_lemma(df):
    # Remove sentences where a lemma is repeated
    # verbs are not included if one verb is trans and the other is intrans
    LEMMAS = get_all_lemmas(Words)
    for lemmas in LEMMAS:
        regex_lemmas = r"|".join([f"\\b{lemma}\\b" for lemma in lemmas])
        doubleE = f"({regex_lemmas}).*({regex_lemmas})"
        mask = df["sentence"].str.contains(doubleE)
        df = df[~mask]
    df = df.reset_index(drop=True)
    return df


def remove_impossible_binding(df):
    # remove "He saw him", "I saw me", etc.
    if set(
        ["subj_type", "obj_type", "obj_REFL", "incongruence_subj_obj_count"]
    ).issubset(df.columns):
        print(df.columns)
        binding_problems = (
            (df["subj_type"] == "PRO")
            & (df["obj_type"] == "PRO")
            & (~(df["obj_REFL"] == True))
            & (df["incongruence_subj_obj_count"] == 0)
        )
        for test_exclude in ["I saw me", "She saw her"]:
            assert (test_exclude in df[binding_problems]["sentence"]) or (
                not (test_exclude in df["sentence"])
            )
        for test_include in ["I saw myself", "She saw herself"]:
            assert not (test_include in df[binding_problems]["sentence"])
        df = df[~binding_problems]
    df = df.reset_index(drop=True)
    return df


def calc_incongruence_counts(df):
    if set(["poss", "subj"]).issubset(df.columns):
        df = calc_incongruence_count(df, "poss", "subj")
    if set(["poss", "obj"]).issubset(df.columns):
        df = calc_incongruence_count(df, "poss", "obj")
    if set(["subj", "obj"]).issubset(df.columns):
        df = calc_incongruence_count(df, "subj", "obj")
    if set(["subj", "embedsubj"]).issubset(df.columns):
        df = calc_incongruence_count(df, "subj", "embedsubj")
    return df


def calc_incongruence_count(df, role1, role2, features=None):
    if features is None:
        features = ["NUM", "GEN", "PERS", "ANIM"]
    df = calc_incongruence_feature(df, role1, role2)
    cols = [f"incongruent_{role1}_{role2}_{feat}" for feat in features]
    cols = [c for c in cols if c in df.columns]
    df[f"incongruence_{role1}_{role2}_count"] = df[cols].sum(axis=1, min_count=1)
    return df


def calc_incongruence_feature(df, role1, role2):
    for feat in ["NUM", "GEN", "PERS", "ANIM"]:
        if f"{role1}_{feat}" in df.columns and set(
            [f"{role2}_{feat}", "poss_type"]
        ).issubset(df.columns):
            df[f"incongruent_{role1}_{role2}_{feat}"] = df.apply(
                lambda row: check_incongruence(
                    row[f"{role1}_{feat}"],
                    row[f"{role2}_{feat}"],
                    role1,
                    role2,
                    row["poss_type"],
                ),
                axis=1,
            )
    return df


def add_sentence_length(df):
    df["sentence_length"] = df.apply(lambda row: len(row["sentence"].split()), axis=1)
    return df


def add_has_property(df, groups=["subjrel", "objrel", "embed", "main"]):
    for group in groups:
        if f"has_{group}" and "sentence_GROUP" in df:
            df[f"has_{group}"] = df.apply(
                lambda row: row["sentence_GROUP"].startswith(f"{group}"), axis=1
            )

    if set(["has_subjrel", "has_objrel", "has_relative_clause"]).issubset(df.columns):
        df["has_relative_clause"] = df.apply(
            lambda row: (row["has_subjrel"] or row["has_objrel"]), axis=1
        )
    return df


def sanity_checks(sentence, tree):
    for fragment_test in [
        "dog see ",  # not a perfect test: "The boy that saw the dogs see the man"
        "dogs falls",  # not a perfect test: "The boy that saw the dogs falls"
    ]:
        if fragment_test in sentence:
            print("WARNING")
            print(sentence)
            print(tree)
        return


def compute_mean_zipf(
    row,
    words,
    lang="en",
):
    zipfs = []
    for word in words:
        if not pd.isnull(row[word]):
            zipfs.append(wordfreq.zipf_frequency(row[word], lang=lang))
    return np.floor(np.mean(zipfs))


def add_word_zipf(df):
    words = [
        w
        for w in df.columns
        if w
        in ["subj", "embedsubj", "verb", "obj", "embedsubj", "embedverb", "matrixverb"]
    ]
    df["mean_zipf"] = df.apply(lambda row: compute_mean_zipf(row, words, "en"), axis=1)
    return df


def get_clause_type(row):
    if row["sentence_GROUP"].startswith(("main_", "embed_")):
        # remove prefix (embed_/main_) and suffix (_clause) from sentence_GROUP
        return row["sentence_GROUP"].split("_")[1]
    else:
        return np.nan


def add_clause_type(df):
    if "clause_type" in df:
        df["clause_type"] = df.apply(lambda row: get_clause_type(row), axis=1)
    return df


def get_aux_tense(row):
    if "sentence_GROUP" in row:
        if row["sentence_GROUP"].startswith("main_obj"):
            return row["do_TENSE"]
    else:
        return row["verb_TENSE"]


def modify_tense_of_obj_quest(df):
    if "verb_TENSE" in df:
        df["verb_TENSE"] = df.apply(lambda row: get_aux_tense(row), axis=1)
    return df


def check_lr_attractor(row):
    if "sentence_GROUP" in row:
        if row["sentence_GROUP"] in ["pp", "objrel"]:
            return check_incongruence(row["subj_NUM"], row["embedsubj_NUM"])
        elif row["sentence_GROUP"] == "subjrel":
            return check_incongruence(row["subj_NUM"], row["embedobj_NUM"])
        else:
            return np.nan
    else:
        return np.nan


def lr_agreement_with_attractor(df):
    df["long_range_agreement_with_attractor"] = df.apply(
        lambda row: check_lr_attractor(row), axis=1
    )
    return df


def calc_simple_binding(row):
    bound_variable = np.nan
    coref_variable = np.nan
    """
    if 'quantifier' not in row: # TO VERIFY
        return {
        'bound_variable': bound_variable,
        'coref_variable': coref_variable}
    """
    if "quantifier" in row and pd.isnull(row["quantifier"]):
        if row["subj_type"] == "PRO":
            coref_variable = row["obj_REFL"] == True
        elif "poss_type" in row and (row["poss_type"] == "subj"):
            agreement_mismatch = get_agreement_mismatch(row, "poss", "obj")
            coref_variable = not agreement_mismatch["overall"]
        elif row["obj_type"] == "PRO":
            coref_variable = row["obj_REFL"] == True
        elif "poss_type" in row and (row["poss_type"] == "obj"):
            agreement_mismatch = get_agreement_mismatch(row, "poss", "subj")
            coref_variable = not agreement_mismatch["overall"]
    elif "quantifier" in row and not (pd.isnull(row["quantifier"])):
        if row["obj_type"] == "PRO":
            bound_variable = row["obj_REFL"] == True
        elif "poss_type" in row and (row["poss_type"] == "obj"):
            agreement_mismatch = get_agreement_mismatch(row, "poss", "subj")
            bound_variable = not agreement_mismatch["overall"]
    return {"bound_variable": bound_variable, "coref_variable": coref_variable}


def add_binding(df):
    binding_cols = df.apply(calc_simple_binding, axis=1, result_type="expand")
    df = pd.concat([df, binding_cols], axis=1)
    return df
