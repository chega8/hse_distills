"""Single-model benchmark; use separate processes for baseline and quantized weights."""
import argparse
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--out', required=True)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
model, tokenizer = load(a.model)
cases = {
    'short': 'Объясни простыми словами, как работает кеш в компьютере. Приведи пример.',
    'long': ('В компании обрабатывают обращения клиентов. Важны скорость ответа, точность и защита данных. ' * 32) + '\nПредложи подробный план улучшения службы поддержки.',
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, 'w') as f:
    for name, prompt in cases.items():
        tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        for repeat in range(-1, a.repeats):
            mx.clear_cache()
            mx.reset_peak_memory()
            start = time.perf_counter()
            first = None
            answer = ''
            for r in stream_generate(model, tokenizer, tokens, max_tokens=128, sampler=make_sampler(temp=0)):
                if first is None:
                    first = time.perf_counter() - start
                answer += r.text
            mx.synchronize()
            row = dict(model=a.model, case=name, repeat=repeat, warmup=repeat == -1,
                       ttft_s=first, total_s=time.perf_counter()-start,
                       prompt_tokens=r.prompt_tokens, prompt_tps=r.prompt_tps,
                       generation_tokens=r.generation_tokens, generation_tps=r.generation_tps,
                       peak_mlx_gb=r.peak_memory, answer=answer)
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
            f.flush()
            print(json.dumps({k:v for k,v in row.items() if k != 'answer'}), flush=True)
