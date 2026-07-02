# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) connects a language model to an external
knowledge source. Instead of relying on whatever the model memorized during
training, the system first retrieves passages relevant to the user's question
and then instructs the model to answer using those passages. The model becomes
a reader and synthesizer rather than an oracle.

## Why RAG exists

Language models have three structural problems that RAG addresses. First,
knowledge cutoff: a model trained in 2024 knows nothing about 2025. Second,
hallucination: when a model lacks knowledge, it generates plausible-sounding
fabrications rather than admitting ignorance. Third, private data: no public
model has read your company's internal wiki or your personal notes. Retrieval
solves all three by injecting current, verifiable, private context directly
into the prompt at question time.

## Grounding and citations

An answer is grounded when every claim it makes is supported by the retrieved
passages. The standard technique numbers each passage in the prompt and
instructs the model to cite passage numbers next to each claim, producing
answers like "The k1 parameter controls term frequency saturation [2]."
Citations serve two purposes: the reader can verify claims against sources, and
the instruction itself measurably reduces hallucination, because the model is
pushed to attend to the provided context instead of its parametric memory. A
well-prompted RAG system must also refuse: when the retrieved passages do not
contain the answer, the correct output is "the provided context does not answer
this question," not a confident guess.

## Chunking strategy

Documents are split into chunks before indexing because retrieval granularity
matters: retrieving a whole book tells you nothing about where the answer is.
Chunk size is a genuine tradeoff. Small chunks produce precise embeddings but
strip away surrounding context; large chunks preserve context but blur multiple
topics into one vector. Overlapping windows — repeating the last sentences of
each chunk at the start of the next — protect against answers that straddle a
boundary. There is no universally correct chunk size; rigorous systems measure
retrieval quality across sizes rather than guessing.

## Evaluating retrieval quality

Retrieval evaluation asks: given a question, did the system surface the passage
that answers it? Recall@k measures the fraction of test questions whose correct
passage appears in the top k results. Mean Reciprocal Rank (MRR) is stricter
about position: each question scores 1 divided by the rank of the correct
passage, so rank 1 earns 1.0, rank 4 earns 0.25, and missing entirely earns 0.
MRR rewards systems that put the right passage first, which matters because
language models attend most reliably to the beginning of their context.

Test questions can be generated synthetically: sample a chunk, ask a language
model to write a question that this specific chunk answers, and record the
chunk's identifier as ground truth. A few dozen such pairs are enough to
compare retrieval strategies with real numbers instead of anecdotes.

## Maximal Marginal Relevance

Top-k retrieval often returns five near-duplicate passages, wasting context
window on redundancy. Maximal Marginal Relevance (MMR) selects passages
greedily: each pick maximizes a weighted combination of relevance to the query
and dissimilarity to passages already selected. The lambda parameter controls
the balance — 1.0 is pure relevance, 0.0 is pure diversity. Values around 0.7
keep answers on-topic while covering more distinct aspects of the question.
