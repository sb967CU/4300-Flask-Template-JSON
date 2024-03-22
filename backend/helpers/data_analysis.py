from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer

def filler_words(data):
    res = set()
    
    for gym_id, gym_info in data.items():
        tokens = TreebankWordTokenizer().tokenize(f"{gym_info['name']} {gym_info['description']} {' '.join(gym_info['reviews'])}")
        for word,count in Counter(tokens).items():
            if count > 10 and count > 0.015 * len(tokens):
                res.add(word)
    return res

def build_inverted_index(data:dict):
  """
  returns dict(word:gym_id)
  """

  res = defaultdict(set)

  for gym_id, gym_info in data.items():
    combined_text = TreebankWordTokenizer().tokenize(f"{gym_info['name']} {gym_info['description']} {' '.join(gym_info['reviews'])}")

    for word in combined_text:
      res[word].add(gym_id)

  return res
