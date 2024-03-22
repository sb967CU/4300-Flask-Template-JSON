from collections import defaultdict, Counter
from nltk.tokenize import TreebankWordTokenizer

def build_filler_words(data):
    res = set()
    
    s = []
    for gym_id, gym_info in data.items():
        s += TreebankWordTokenizer().tokenize(f"{gym_info['name']} {gym_info['description']} {' '.join(gym_info['reviews'])}")
    for word,count in Counter(s).items():
      if count > 10 and count > 0.015 * len(s):
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
