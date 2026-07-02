# BM25 and Sparse Retrieval

Sparse retrieval represents documents by the words they actually contain. The
name comes from the vectors involved: a document maps to a vocabulary-sized
vector that is almost entirely zeros, with non-zero weights only at positions
corresponding to its words. BM25 is the dominant scoring function of this
family and remains the baseline to beat in information retrieval, four decades
of neural progress notwithstanding.

## From TF-IDF to BM25

TF-IDF scores a term in a document as term frequency (how often the word occurs
in this document) multiplied by inverse document frequency (how rare the word is
across all documents). The intuition: a word matters if it is common in this
document but uncommon in general. The word "the" appears everywhere, so its IDF
is near zero; the word "photosynthesis" is rare, so documents containing it get
strong signal.

BM25 refines term frequency with two corrections. First, saturation: the k1
parameter caps the benefit of repetition, so a word appearing 50 times does not
score ten times higher than appearing 5 times — relevance does not scale
linearly with repetition. Second, length normalization: the b parameter
penalizes long documents, because a 10,000-word document mentioning your query
term once is weaker evidence than a 100-word document doing the same. Typical
values are k1 between 1.2 and 2.0, and b around 0.75.

## Where BM25 beats dense retrieval

BM25 excels exactly where embeddings fail: exact identifiers, code symbols,
version numbers, proper nouns, domain jargon. A query for "IndexFlatIP" will
retrieve documents containing that literal token with certainty, while a dense
model may consider it just another technical-looking string. BM25 also needs no
training, no GPU, and behaves predictably — a new term is searchable the moment
it is indexed.

## Where BM25 fails

BM25 has zero understanding of meaning. It cannot connect "car" with
"automobile" or "reset password" with "forgot login". Vocabulary mismatch
between how a user phrases a question and how a document phrases the answer is
the fundamental weakness of all sparse methods, and it is the single strongest
argument for dense retrieval.

## Rank fusion: combining sparse and dense

BM25 scores and cosine similarities live on incompatible scales, so adding them
directly is meaningless. Reciprocal Rank Fusion (RRF) sidesteps this by
discarding scores entirely and combining ranks: each document receives
1/(k + rank) from every result list it appears in, with k commonly set to 60.
Documents ranked highly by both systems accumulate the largest fused scores.
RRF is embarrassingly simple, has essentially one parameter, and is remarkably
hard to beat — production search systems at major companies still rely on it.

## The two-stage retrieval pattern

Practical systems retrieve in two stages. Stage one casts a wide net cheaply:
BM25 and dense search each return their top candidates, fused by RRF. Stage two
spends expensive computation on precision: a cross-encoder reranker jointly
reads the query and each candidate to produce a sharper relevance score. This
funnel — cheap recall, then expensive precision — is the same pattern used by
web search engines and recommendation systems at every scale.
