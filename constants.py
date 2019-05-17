

class Constants():
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    PAD = "<PAD>"
    index2special_token = {
        0: PAD,
        1: SOS,
        2: EOS,
        3: UNK
    }
    special_token2index = {v: k for k, v in index2special_token.items()}
    print(special_token2index.items())