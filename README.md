## Create .json file with prompts
- `create_prompts.ipynb`

## Send prompts to model
```bash
python assync_send_prompts.py --base_url https://openrouter.ai/api/v1 --api_key <OPENROUTER_API_KEY> --prompts_path "prompts/ptpt-prompts.json" --responses_path "results/gpt-4o-ptpt-responses.json" --model openai/gpt-4o --concurrency 8
```

## Evaluate model answers
- `evaluate-answers.ipynb`