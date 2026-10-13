"""Ten balanced shards, five examples per task; one method per GPU worker."""
from collections import Counter


def work_plan(config, manifest):
    shards=config['formal']['shards']
    items=manifest['evaluation']
    tasks=sorted({r['task'] for r in items})
    buckets=[[] for _ in range(shards)]
    loads=[0]*shards
    for task in tasks:
        subset=sorted((r for r in items if r['task']==task),key=lambda r:(-r['input_tokens']**2,r['key']))
        if len(subset)%shards:
            raise ValueError('task count must divide evenly across shards')
        capacity=len(subset)//shards
        counts=[0]*shards
        for item in subset:
            shard=min((i for i in range(shards) if counts[i]<capacity),key=lambda i:(loads[i],i))
            buckets[shard].append(item)
            counts[shard]+=1
            loads[shard]+=item['input_tokens']**2
    result=[]
    for shard,bucket in enumerate(buckets):
        keys=[r['key'] for r in sorted(bucket,key=lambda r:(-r['input_tokens']**2,r['key']))]
        if len(keys)!=config['formal']['samples_per_worker']:
            raise ValueError('invalid worker sample count')
        for method in config['methods']:
            result.append(dict(worker=len(result),shard=shard,method=method,keys=keys,
                task_counts=dict(Counter(r['task'] for r in bucket)),quadratic_load=loads[shard]))
    return result
