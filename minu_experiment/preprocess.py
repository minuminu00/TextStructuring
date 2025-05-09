import os
import nltk
import pathlib

# __path__
# pathlib

path = pathlib.Path('./TextStructuring/minu_experiment/wiki_727K/wiki_727/dev')
files = [____path for _path in path.glob('**/*') for __path in _path.glob('**/*') for ___path in __path.glob('**/*') for ____path in ___path.glob('**/*')]

totalFile = len(files)
files = files[:totalFile//10]

texts = []

class Sentence:
    def __init__(self, raw: str):
        self.raw = raw
        self.tokens = nltk.tokenize.word_tokenize(raw)
    def __len__(self):
        return len(self.tokens)

class Paragraph:
    def __init__(self, raw: str):
        self.raw = raw
        self.sentences = [Sentence(sent)
                          for sent in nltk.tokenize.sent_tokenize(raw)]
    def __len__(self):
        return len(self.sentences)

class Text:
    def __init__(self, lines: list[str], train: bool):
        assert "***LIST***.\n" not in lines
        
        self.raw = " ".join([line for line in lines if line[:9] != "========,"])

        paragraphs = []
        currParagraph = ""

        for line in lines:
            if line[:9] == "========,":
                if currParagraph:
                    paragraphs.append(Paragraph(currParagraph))
                    currParagraph = ""
            else:
                if currParagraph:
                    currParagraph += " "
                currParagraph += line[:-1]
        if currParagraph:
            paragraphs.append(Paragraph(currParagraph))
        
        self.paragraphs = paragraphs
        
        self.sentences = [sentence
                          for paragraph in paragraphs
                          for sentence in paragraph.sentences]
        
        self.ans = [0]
        for paragraph in paragraphs:
            self.ans.append(self.ans[-1] + len(paragraph))
        self.ans = self.ans[:-1]

def get_cfd_headword(texts: list[Text]):
    return nltk.ConditionalFreqDist(
        [("all", token.lower())
        for text in texts
        for paragraph in text.paragraphs
        for sentence in paragraph.sentences
        for token in sentence.tokens] +
        [("head-all", token.lower())
        for text in texts
        for paragraph in text.paragraphs
        for sentence in paragraph.sentences
        for token in sentence.tokens[:3]] +
        [("headwords", token.lower())
        for text in texts
        for paragraph in text.paragraphs
        for token in paragraph.sentences[0].tokens] +
        [("head-headwords", token.lower())
        for text in texts
        for paragraph in text.paragraphs
        for token in paragraph.sentences[0].tokens[:3]]
    )

def get_collocation(texts: list[Text]):
    cfd = nltk.ConditionalFreqDist()
    for text in texts:
        for paragraph in text.paragraphs:
            tokens = [token for sentence in paragraph.sentences
                      for token in sentence.tokens]
            for token1 in tokens:
                for token2 in tokens:
                    _token1 = token1.lower()
                    _token2 = token2.lower()
                    if _token1 != _token2:
                        cfd[token1][token2] += 1
    fd = nltk.FreqDist(
        token for text in texts
        for paragraph in text.paragraphs
        for sentence in paragraph.sentences
        for token in sentence.tokens
    )
    return cfd, fd
    

def split_bayesian(cfd_headword, text: Text):
    """Bayesian"""
    defaultProb = sum(cfd_headword["head-all"].values()) / sum(cfd_headword["all"].values())
    score = []
    for sentence in text.sentences:
        prob = []
        for token in sentence.tokens:
            occ_head = cfd_headword["head-all"][token.lower()]
            occ_whole = cfd_headword["all"][token.lower()]
            prob.append(((occ_head + defaultProb * 1000) / (occ_whole+1000)))
        """for token in sentence.tokens[:3]:
            occ_head = cfd_headword["head-headwords"][token.lower()]
            occ_whole = cfd_headword["headwords"][token.lower()]
            if occ_whole > 30 and abs((occ_head / occ_whole) - defaultProb) > 0.03:
                prob.append(occ_head / occ_whole)"""
        score.append(sum(prob) / len(prob)
                     if prob
                     else defaultProb)
    return []


texts = []

for file in files:
    f = open(file, 'r', encoding='utf8')
    lines = f.readlines()
    if "***LIST***.\n" in lines: continue
    texts.append(Text(lines, True))

print("Total file #:", len(files))
print("Valid file #:", len(texts))

cfd_headword = get_cfd_headword(texts)
cfd_collocation, fd_word = get_cfd_headword(texts)

for i in range(20):
    print(texts[i].ans)
    print(split_bayesian(cfd_headword, texts[i]))