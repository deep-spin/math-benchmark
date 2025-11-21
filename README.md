## Create .json file with prompts
- `create_prompts.ipynb`

## Send prompts to model

- Create a .env file with the environment variable <OPEN_ROUTER_API_KEY>

- Send prompts to closed models using OpenRouter API key 
```bash
python assync_send_prompts.py --base_url https://openrouter.ai/api/v1 --api_key openrouter --prompts_path "prompts/ptpt-open-ended-prompts-prompt-language-ptpt.json" --responses_path "results/open-ended/qwen3-14b-ptpt-open-ended-responses-prompt-language-ptpt.json" --model qwen/qwen3-14b --concurrency 8
```


- Send prompts to local models using vLLM OpenAI-Compatible Server
```bash
python assync_send_prompts.py --base_url http://localhost:8000/v1 --api_key token-abc123 --prompts_path "prompts/ptpt-prompts.json" --responses_path "results/Qwen3-4B-ptpt-responses.json" --model /mnt/d/models/Qwen3-4B --concurrency 8
```

## Evaluate model answers

### Multiple-Choice Questions
- `evaluate-multiple-choice-answers.ipynb`

### Open-Ended Questions
- `create_judge_prompts.ipynb`
- Send Judge Prompts to Judge
```bash
python assync_send_prompts.py --base_url https://openrouter.ai/api/v1 --api_key openrouter --prompts_path "judge-prompts/qwen3-14b-ptpt-judge-prompts-prompt-language-ptpt.json" --responses_path "judge-results/qwen3-14b-ptpt-judge-responses-prompt-language-ptpt-judge-qwen3-14b.json" --model qwen/qwen3-14b --concurrency 8
```
- `evaluate-open-ended-answers.ipynb`