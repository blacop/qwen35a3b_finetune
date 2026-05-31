#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path


def stable_split(key: str) -> str:
    v = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % 10
    if v == 0:
        return 'val'
    if v == 1:
        return 'test'
    return 'train'


def first_turn_pair(messages):
    user = None
    for m in messages:
        role = m.get('role')
        content = (m.get('content') or '').strip()
        if not content:
            continue
        if role == 'user' and user is None:
            user = content
        elif role == 'assistant' and user:
            return user, content
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=20260407)
    args = ap.parse_args()

    random.seed(args.seed)

    inp = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft = out / 'sft_openai_messages.jsonl'
    dpo = out / 'dpo_pairs.jsonl'
    grpo = out / 'grpo_prompts.jsonl'

    bad_answers = [
        '请稍后再试。',
        '这个问题我不清楚。',
        '请联系其他客服。',
        '无法处理。',
    ]

    n_in = 0
    n_sft = 0
    n_dpo = 0
    n_grpo = 0

    with inp.open('r', encoding='utf-8') as fin, \
            sft.open('w', encoding='utf-8') as fsft, \
            dpo.open('w', encoding='utf-8') as fdpo, \
            grpo.open('w', encoding='utf-8') as fgrpo:

        for line in fin:
            n_in += 1
            obj = json.loads(line)
            did = obj.get('dialogue_hash') or obj.get('id') or str(n_in)
            split = stable_split(did)
            messages = obj.get('messages', [])
            if not messages:
                continue

            # SFT: full multi-turn messages
            sft_row = {
                'id': obj.get('id', f'row_{n_in}'),
                'split': split,
                'messages': messages,
                'meta': obj.get('meta', {}),
            }
            fsft.write(json.dumps(sft_row, ensure_ascii=False) + '\n')
            n_sft += 1

            # DPO/GRPO: use first user->assistant pair
            prompt, chosen = first_turn_pair(messages)
            if not prompt or not chosen:
                continue

            dpo_row = {
                'id': f"dpo_{obj.get('id', n_in)}",
                'split': split,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': random.choice(bad_answers),
                'meta': {'source_dialogue_id': obj.get('id'), 'source_hash': did},
            }
            fdpo.write(json.dumps(dpo_row, ensure_ascii=False) + '\n')
            n_dpo += 1

            grpo_row = {
                'id': f"grpo_{obj.get('id', n_in)}",
                'split': split,
                'messages': [{'role': 'user', 'content': prompt}],
                'reference_answer': chosen,
                'reward_rubric': {
                    'intent_match': 0.35,
                    'policy_compliance': 0.30,
                    'actionability': 0.20,
                    'clarity': 0.15,
                },
                'meta': {'source_dialogue_id': obj.get('id'), 'source_hash': did},
            }
            fgrpo.write(json.dumps(grpo_row, ensure_ascii=False) + '\n')
            n_grpo += 1

    print(json.dumps({
        'input_rows': n_in,
        'sft_rows': n_sft,
        'dpo_rows': n_dpo,
        'grpo_rows': n_grpo,
        'outputs': [str(sft), str(dpo), str(grpo)],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
