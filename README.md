# U1MIndexer

An indexer that leverages [hnswlib](https://github.com/nmslib/hnswlib) for vector search and `DocumentArrayMemmap` for storing full Documents. `U1M` means under one million Documents. It is a perfect indexer if you are working with indexers less than 1M and aims for query-time speed. When working with one million Documents, you can expect single query to be complete at 2ms and 600~800 QPS on batch queries. When working with more than 1M Documents, one has to be careful with its memory consumption. The indexing speed of `U1MIndexer` is much slower comparing to [U100KIndexer](https://hub.jina.ai/executor/80scarrt).

`U1MIndexer` returns approximate nearest neighbours.

The code is based on [HNSWSearcher](https://hub.jina.ai/executor/jdb3vkgo)

## Pros & cons

### Pros

- Extremely fast query speed, irrelevant to the number of stored Documents.
- Always return full Documents.

### Cons

- Indexing is relatively slow.
- Extra dependencies: `hnswlib` and `bidict`.
- Extra parameters to tune HNSW performance.

## Performance

One can run `benchmark.py` to get a quick performance overview. 

|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|10000 | 2.903 | 0.002 | 0.013 | 0.092|
|100000 | 65.330 | 0.002 | 0.014 | 0.104|
|500000 | 463.509 | 0.003 | 0.019 | 0.132|
|1000000 | 988.535 | 0.003 | 0.017 | 0.127|
