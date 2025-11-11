## Create .json file with prompts
- `create_prompts.ipynb`

## Send prompts to model

- Create a .env file with the environment variable <OPEN_ROUTER_API_KEY>

- Send prompts to closed models using OpenRouter API key 
```bash
python assync_send_prompts.py --base_url https://openrouter.ai/api/v1 --api_key openrouter --prompts_path "prompts/ptpt-prompts.json" --responses_path "results/gpt-5-ptpt-responses.json" --model openai/gpt-5 --concurrency 8
```


- Send prompts to local models using vLLM OpenAI-Compatible Server
```bash
python assync_send_prompts.py --base_url http://localhost:8000/v1 --api_key token-abc123 --prompts_path "prompts/ptpt-prompts.json" --responses_path "results/Qwen3-4B-ptpt-responses.json" --model /mnt/d/models/Qwen3-4B --concurrency 8
```

## Evaluate model answers
- `evaluate-answers.ipynb`