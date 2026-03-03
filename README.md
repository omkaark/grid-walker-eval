# grid-walker-eval

Save your OPENROUTER_API_KEY in .env

To eval a model from openrouter:
```bash
python -m src.eval.main --model bytedance-seed/seed-1.6-flash --seeds 0 --verbose --grid-size 16 --blocks 10 --log-images bytedance_seed
```

or for eval w/ vllm inference:
```bash
python -m src.eval.main --model Qwen/Qwen3-VL-2B-Instruct --base-url http://localhost:8000/v1 --api-key none --log-images test --seeds 0,1,2,3,4,5,6,7,8,9 --blocks 3 --grid-size 8
```

Start vLLM:
```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-VL-2B-Instruct --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.95 --enable-sleep-mode --enable-lora --lora-modules policy=adapter_policy/rl
```

Generate synthetic data:
```bash
python -m src.data.generate --n-blocks 3 --grid-size 8 --samples 1000 --workers 10
```

Do finetune for midtrain:
```bash
python -m src.train.finetune --dataset-dir dataset/simpleds --output-dir adapter_policy --epochs 1 --batch-size 2 --grad-accum 8 --lr 1e-5 --max-turns-per-sample 6 --max-seq-len 110246 --batch-log-every 1 --log-every 1 --save-every-updates 1
```

Do RL:
```bash
GRID_WALKER_MAX_TURNS=10 GRID_WALKER_N_STEPS=3 MODEL_NAME=gw ADAPTER_PATH="./adapter_policy/rl" python -m src.train.rl
```