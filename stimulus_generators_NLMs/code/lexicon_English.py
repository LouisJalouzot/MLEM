def keep_first_k_elements(d, first_k=1):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # if value is a dictionary, recursively call the function
            new_dict[k] = keep_first_k_elements(v, first_k)
        elif isinstance(v, list):
            # if value is a list, keep only the first element
            new_dict[k] = v[:first_k]
        else:
            # for other types of values, just copy the value as is
            new_dict[k] = v
    return new_dict


###################
###### QUANTIFIERS ######
###################
# Initialization
quantifiers = {}
# Tokens
quantifiers["singular"] = ["every", "no"]
quantifiers["plural"] = ["all", "few"]

###################
###### PROPER NAMES ######
###################
# Initialization
proper_names = {}
for number in ["singular"]:
    proper_names[number] = {}
# Tokens
proper_names["singular"]["masculine"] = ["John", "Bob", "Lex"]
proper_names["singular"]["feminine"] = ["Mary", "Patricia", "Lori"]

###################
###### NOUNS ######
###################
# Initialization
nouns = {}
for gender in ["masculine", "feminine"]:
    nouns[gender] = {}
# Tokens
nouns["masculine"]["singular"] = ["man", "prince", "actor"]  # steward, waiter,
nouns["masculine"]["plural"] = ["men", "princes", "actors"]
nouns["feminine"]["singular"] = ["woman", "princess", "actress"]
nouns["feminine"]["plural"] = ["women", "princesses", "actresses"]


nouns_inanimate = {}
nouns_inanimate["singular"] = ["book", "plate", "pencil"]
nouns_inanimate["plural"] = ["books", "plates", "pencils"]


###################
###### VERBS ######
###################
# Initialization
verbs, verbs_intran_anim, verbs_intran_inanim, matrix_verbs, do_Aux = {}, {}, {}, {}, {}
for tense in ["present"]:
    (
        verbs[tense],
        verbs_intran_anim[tense],
        verbs_intran_inanim[tense],
        matrix_verbs[tense],
        do_Aux[tense],
    ) = ({}, {}, {}, {}, {})

verbs["past"] = ["saw", "held", "admired"]
verbs["present"]["singular"] = ["sees", "holds", "admires"]
verbs["present"]["plural"] = ["see", "hold", "admire"]
verbs["future"] = ["will see", "will hold", "will admire"]
verbs["-finite"] = verbs["present"]["plural"]

verbs_intran_anim["past"] = ["played", "sang", "sneezed"]
verbs_intran_anim["present"]["singular"] = ["plays", "sings", "sneezes"]
verbs_intran_anim["present"]["plural"] = ["play", "sing", "sneeze"]
verbs_intran_anim["future"] = ["will play", "will sing", "will sneeze"]

verbs_intran_inanim["past"] = ["fell", "disappeared", "vanished"]
verbs_intran_inanim["present"]["singular"] = ["falls", "disappears", "vanishes"]
verbs_intran_inanim["present"]["plural"] = ["fall", "disappear", "vanish"]
verbs_intran_inanim["future"] = ["will fall", "will disappear", "will vanish"]

# matrix_verbs['past'] =                ['knew', 'remembered', 'declared']
matrix_verbs["present"]["singular"] = ["knows", "remembers", "announces"]
matrix_verbs["present"]["plural"] = ["know", "remember", "announce"]
matrix_verbs["future"] = ["will know", "will remember", "will announce"]
# matrix_verbs['-finite'] = matrix_verbs['present']['plural']

copula = {}
copula["singular"] = ["is"]
copula["plural"] = ["are"]

do_Aux["past"] = ["did"]
do_Aux["present"]["singular"] = ["does"]
do_Aux["present"]["plural"] = ["do"]
do_Aux["future"] = ["will"]

##########################
###### PREPOSITIONS ######
##########################

# LOCATION PREPOSITIONS
loc_preps = ["near", "behind"]


Words = {
    "nouns": nouns.copy(),
    "nouns_inanimate": nouns_inanimate.copy(),
    "verbs": verbs.copy(),
    "verbs_intran_anim": verbs_intran_anim.copy(),
    "verbs_intran_inanim": verbs_intran_inanim.copy(),
    "copula": copula.copy(),
    "do_Aux": do_Aux.copy(),
    "matrix_verbs": matrix_verbs.copy(),
    "loc_preps": loc_preps.copy(),
    "quantifiers": quantifiers.copy(),
    "proper_names": proper_names.copy(),
}


def get_Words(n):
    res = keep_first_k_elements(Words, n)
    return res


if __name__ == "__main__":
    pass
