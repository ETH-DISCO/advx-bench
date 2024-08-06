# python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")
sentence = "a photo of a cute little kitten laying on a blanket next to a dog laying on a bed"
doc = nlp(sentence)
noun_chunks = [chunk.text for chunk in doc.noun_chunks]

print(noun_chunks)
