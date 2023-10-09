

def convert_ptp_to_universal(ptp_tag):
    struct = {
        "#": "SYM",
        "$": "SYM",
        "''": "PUNCT",
        "(": "PUNCT",
        ")": "PUNCT",
        ",": "PUNCT",
        ".": "PUNCT",
        ":": "PUNCT",
        "``": "PUNCT",
        "CC": "CCONJ",
        "CD": "NUM",
        "DT": "DET",
        "EX": "PRON",
        "FW": "X",
        "IN": "ADP",
        "JJ": "ADJ",
        "JJR": "ADJ",
        "JJS": "ADJ",
        "LS": "X",
        "MD": "VERB",
        "NN": "NOUN",
        "NNS": "NOUN",
        "NNP": "PROPN",
        "NNPS": "PROPN",
        "PDT": "DET",
        "POS": "PART",
        "PRP": "PRON",
        "PRP$": "DET",
        "RB": "ADV",
        "RBR": "ADV",
        "RBS": "ADV",
        "RP": "ADP",
        "SYM": "SYM",
        "TO": "PART",
        "UH": "INTJ",
        "VB": "VERB",
        "VBD": "VERB",
        "VBG": "VERB",
        "VBN": "VERB",
        "VBP": "VERB",
        "VBZ": "VERB",
        "WDT": "DET",
        "WP": "PRON",
        "WP$": "DET",
        "WRB": "ADV"
    }
    
    if ptp_tag in struct:
        return struct[ptp_tag]
    else:
        return ptp_tag

def load_conll2000_file(path):
    tagged_sentences = []
    sent = []
    
    with open(path, "r") as file:
        for x in file:
            if x != "\n":
                elements = x.split(" ")
                
                if elements[0] == "-LRB-" or elements[0] == "-LCB-":
                    elements[0] = "("
                elif elements[0] == "-RRB-" or elements[0] == "-RCB-":
                    elements[0] = ")"
                
                tag = convert_ptp_to_universal(elements[1])
                word = elements[0]
                    
                sent.append((tag, word))
            else:
                tagged_sentences.append(sent)
                sent = []
    
    return tagged_sentences
        
def load_conll2000(path = "./"):
    train_set = load_conll2000_file(path + "train_conll2000.txt")
    test_set = load_conll2000_file(path + "test_conll2000.txt")
    
    return train_set, test_set