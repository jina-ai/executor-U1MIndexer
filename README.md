# U1MIndexer

An indexer that leverages [hnswlib](https://github.com/nmslib/hnswlib) for vector search and `DocumentArrayMemmap` for storing full Documents. `U1M` means under one million Documents. It is a perfect indexer if you are working with indexers less than 1M and aims for query-time speed. The indexing is much slower comparing to [U100KIndexer](https://hub.jina.ai/executor/80scarrt).

The code is based on [HNSWSearcher](https://hub.jina.ai/executor/jdb3vkgo)

## Pros & cons

### Pros

- Fast query speed
- Always return full Documents

### Cons

- Indexing is relatively slow
- Extra dependencies: `hnswlib` and `bidict`
- Extra parameters to tune HNSW performance

## Performance

One can run `benchmark.py` to get a quick performance overview. 


|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|10000 | 2.839 | 0.002 | 0.013 | 0.088|
|100000 | 62.942 | 0.002 | 0.016 | 0.108|
|500000 | 469.240 | 0.002 | 0.018 | 0.126|
|1000000 | 1022.492 | 0.003 | 0.020 | 0.130 |
