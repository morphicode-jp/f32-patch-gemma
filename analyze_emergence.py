"""Quick analysis script for phase_12_emergence.json."""
import json
from collections import Counter

d = json.load(open('phase_12_emergence.json', encoding='utf-8'))

print('=== SUMMARY ===')
for k, v in d['summary'].items():
    print(f'  {k}: {v}')

print('\n=== Generation 5+ events (multi-gen breakthrough!) ===')
for h in d['history']:
    for s in h['snapshots']:
        g = s['behavior'].get('gen_max', 0)
        if g >= 5:
            uid = s['u_id']
            alive = s['behavior'].get('n_alive', 0)
            q = s['quality']
            mut = s['world_params']['mutation_rate']
            print(f"  ep{h['epoch']:2d} u{uid}: gen={g} alive={alive} q={q:+.2f} mut={mut:.3f}")

print('\n=== Peak quality ===')
max_quality = max(max(s['quality'] for s in h['snapshots']) for h in d['history'])
print(f'  PEAK: {max_quality:.2f}')

print('\n=== Selection events ===')
print(f'  Total: {len(d["selection_events"])}')
replaced = Counter(e['replaced_uid'] for e in d['selection_events'])
parents = Counter(e['parent_uid'] for e in d['selection_events'])
print(f'  Replaced: {dict(replaced)}')
print(f'  Parents: {dict(parents)}')
