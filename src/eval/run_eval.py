import os, json
import torch

def run(args):
    try:
        from lm_eval import evaluator
    except Exception as e:
        print("Please `pip install lm-eval`", e); raise SystemExit(1)
    model_args = f"pretrained={args.model_path},dtype=bfloat16"
    if torch.cuda.is_available(): model_args += ",device=cuda"
    print("[Eval] model_args:", model_args)
    if os.path.isdir(args.tasks):
        with open(os.path.join(args.tasks, "tasks.txt")) as f:
            tasks = f.read().strip().split(",")
    elif os.path.isfile(args.tasks):
        with open(args.tasks) as f:
            tasks = f.read().strip().split(",")
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    res = evaluator.simple_evaluate(model="hf", model_args=model_args, tasks=tasks, batch_size="auto", limit=args.limit)
    os.makedirs("outputs/eval", exist_ok=True)
    with open("outputs/eval/results.json","w") as f: json.dump(res,f,indent=2)
    # Build a compact report
    md = ["# Evaluation Report",
          "",
          "| Task | Metric | Score |",
          "|---|---|---|"]
    for k,v in res.get("results",{}).items():
        metric = list(v.keys())[0]
        md.append(f"| {k} | {metric} | {v[metric]:.4f} |")
    with open("outputs/eval/report.md","w") as f: f.write("\n".join(md))
    print("Saved eval to outputs/eval/results.json and report.md")
