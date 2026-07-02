# Word and Sentence Embeddings

An embedding is a learned mapping from discrete symbols — words, sentences,
entire documents — to dense vectors of real numbers. The defining property is
that semantic similarity in language corresponds to geometric proximity in the
vector space. Words like "doctor" and "physician" end up near each other, while
"doctor" and "asteroid" end up far apart.

## Why embeddings replaced one-hot vectors

Before embeddings, text was represented with one-hot vectors: a vocabulary of
50,000 words meant 50,000-dimensional vectors with a single 1 and 49,999 zeros.
Two problems made this untenable. First, the dimensionality is enormous and
almost entirely wasted. Second, and more fundamentally, every pair of distinct
words is equally distant — the representation carries no notion that "cat" and
"kitten" are related. Embeddings solve both: they are compact (typically 100 to
1,000 dimensions) and their geometry encodes meaning.

## How embeddings are trained

The classic insight is the distributional hypothesis: words that appear in
similar contexts have similar meanings. Word2vec (2013) operationalized this by
training a shallow network to predict a word from its neighbors (CBOW) or
neighbors from a word (skip-gram). The network's hidden weights become the
embedding. GloVe took a related route through global co-occurrence statistics.

Modern sentence embedding models such as Sentence-BERT are trained differently:
they fine-tune a transformer with a contrastive objective, pulling embeddings of
paraphrase pairs together and pushing unrelated pairs apart. This makes the
cosine similarity between two sentence vectors directly usable as a semantic
similarity score.

## Cosine similarity

Given two vectors a and b, cosine similarity is the dot product divided by the
product of their magnitudes. It measures the angle between vectors, ignoring
their lengths. A value of 1 means identical direction, 0 means orthogonal
(unrelated), and negative values mean opposing directions. When vectors are
normalized to unit length, cosine similarity reduces to a plain dot product,
which is what fast vector search libraries exploit.

## The bi-encoder architecture

Sentence embedding models are bi-encoders: the query and the document are
encoded independently into vectors, and similarity is computed afterwards as a
dot product. The great advantage is that document vectors can be precomputed
and indexed — at query time only the query needs a forward pass through the
model. The cost is that the model never sees the query and document together,
so it cannot model fine-grained interactions between their words. This is the
gap that cross-encoders and reranking close.

## Limitations of dense embeddings

Dense embeddings struggle with exact lexical matches that carry high
information: product codes, function names, rare proper nouns, error strings.
The embedding of "ERR_CONN_RESET_4012" is not meaningfully different from other
error-like strings, whereas a keyword index matches it exactly. Dense retrieval
also degrades on out-of-domain vocabulary the model never saw during training.
These weaknesses are precisely why hybrid retrieval — combining dense vectors
with sparse keyword methods like BM25 — consistently outperforms either method
alone on heterogeneous real-world corpora.
