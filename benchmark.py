import numpy as np
from jina import DocumentArray
from jina.logging.profile import TimeContext
from docarray.math.helper import top_k as _top_k
from docarray.math.distance import cdist

n_index = [10_000, 100_000, 500_000, 1_000_000]
n_query = [1, 8, 64]
D = 768
R = 5
B = 4096

top_k = 20

from executor import U1MIndexer


def _precision(predicted, relevant, eval_at):
    """
    fraction of retrieved documents that are relevant to the query
    """
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(set(predicted_at_k).intersection(set(relevant)))

    return n_predicted_and_relevant / len(predicted)


def _recall(predicted, relevant, eval_at):
    """
    fraction of the relevant documents that are successfully retrieved
    """
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(set(predicted_at_k).intersection(set(relevant)))
    return n_predicted_and_relevant / len(relevant)


def evaluate(predicts, relevants, eval_at):
    recall = 0
    precision = 0
    for _predict, _relevant in zip(predicts, relevants):
        _predict = np.array([int(x) for x in _predict])
        recall += _recall(_predict, _relevant, top_k)
        precision += _precision(_predict, _relevant, top_k)

    return recall / len(predicts), precision / len(predicts)

times = {}
recalls = {}

for n_i in n_index:
    idxer = U1MIndexer(dim=D, limit=top_k, metric='cosine')
    # build index docs
    i_embs = np.random.random([n_i, D]).astype(np.float32)
    da = DocumentArray.empty(n_i)
    da.embeddings = i_embs
    for i, d in enumerate(da):
        d.id = f'{i}'

    with TimeContext(f'indexing {n_i} docs') as t_i:
        for _batch in da.batch(batch_size=B):
            idxer.index(_batch)

    times[n_i] = {}
    times[n_i]['index'] = t_i.duration

    for n_q in n_query:
        q_embs = np.random.random([n_q, D]).astype(np.float32)
        qa = DocumentArray.empty(n_q)
        qa.embeddings = q_embs

        t_qs = []
        r_qs = []
        for _ in range(R):
            with TimeContext(f'searching {n_q} docs') as t_q:
                idxer.search(qa)
            t_qs.append(t_q.duration)
            # # check if it return the full doc
            # assert qa[0].matches
            # assert qa[0].matches.embeddings.shape
            if n_q == 1:
                dists = cdist(q_embs, i_embs, metric='cosine')
                true_dists, true_ids = _top_k(dists, top_k, descending=False)

                pred_ids = []
                for doc in qa:
                    pred_ids.append([m.id for m in doc.matches])

                recall, precision = evaluate(pred_ids, true_ids, top_k)
                r_qs.append(recall)

        if n_q == 1:
            recalls[n_i] = np.mean(r_qs[1:])
            print(f'recall (n={n_i}): {recalls[n_i]}')

        times[n_i][f'query_{n_q}'] = np.mean(t_qs[1:])  # remove warm-up

    idxer.clear()

print('|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|')
print('|---' * (len(list(times.values())[0]) + 1) + '|')
for k, v in times.items():
    s = ' | '.join(f'{v[vv]:.3f}' for vv in ['index', 'query_1', 'query_8', 'query_64'])
    print(f'|{k} | {s}|')
