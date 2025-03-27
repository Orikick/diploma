from collections import defaultdict

ngrams = defaultdict(int)

def get_ngrams(n=2):
    global ngrams
    return [{"ngram": k, "count": v} for k, v in ngrams.items()]

def get_markov_chain():
    # Заготовка для Марковського ланцюга
    return {"nodes": [], "edges": []}