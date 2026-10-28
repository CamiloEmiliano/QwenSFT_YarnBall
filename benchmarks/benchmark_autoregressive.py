#!/usr/bin/env python3
"""
Phase 2: Free-Running Autoregressive Benchmark & Syntax Verification Suite
YarnBall Qwen2.5-7B Unified SFT Model

Runs free-running token-by-token generation across held-out test data (Tasks A through E),
evaluating AST syntax validity, CIK hallucination rate, causal CoT structure, and latency.
"""

import os
import re
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ---------------------------------------------------------------------------
# AST & Syntactic Validators
# ---------------------------------------------------------------------------

def validate_cypher_syntax(text: str) -> Tuple[bool, str]:
    """
    Validates openCypher syntax structure for property graph queries / statements.
    Checks balanced delimiters, clause ordering, and valid relationship syntax.
    """
    cleaned = text.strip()
    if not cleaned:
        return False, "Empty output"
    
    # Check delimiter balance
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    in_single_quote = False
    in_double_quote = False
    
    for ch in cleaned:
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif not in_single_quote and not in_double_quote:
            if ch in '([{':
                stack.append(ch)
            elif ch in ')]}':
                if not stack or stack[-1] != pairs[ch]:
                    return False, f"Unbalanced delimiter: unexpected '{ch}'"
                stack.pop()
                
    if in_single_quote or in_double_quote:
        return False, "Unclosed string literal"
    if stack:
        return False, f"Unclosed opening delimiter: '{stack[-1]}'"

    # Check for valid openCypher graph triple pattern or query clauses
    has_clause = bool(re.search(r'\b(MATCH|MERGE|RETURN|WHERE|WITH|CREATE|SET)\b', cleaned, re.IGNORECASE))
    has_triple = bool(re.search(r'\(:?\w*(\s*\{[^}]*\})?\)-(\[.*?\])?->\(:?\w*(\s*\{[^}]*\})?\)', cleaned))
    
    if not (has_clause or has_triple):
        return False, "Missing valid Cypher clause (MATCH/MERGE/RETURN) or graph triple pattern"
        
    return True, "Valid"


def validate_cik_integrity(prediction: str, ground_truth: str) -> Tuple[bool, List[str], List[str]]:
    """
    Extracts 10-digit CIK numbers and evaluates against ground truth.
    Returns (has_no_hallucinations, predicted_ciks, ground_truth_ciks).
    """
    pred_ciks = set(re.findall(r'\b\d{10}\b', prediction))
    gt_ciks = set(re.findall(r'\b\d{10}\b', ground_truth))
    
    # Check if model invented CIKs that do not appear in ground truth
    hallucinated = pred_ciks - gt_ciks
    return len(hallucinated) == 0, list(pred_ciks), list(gt_ciks)


def validate_reasoning_cot(prediction: str) -> Tuple[bool, str]:
    """
    Validates Chain-of-Thought (<think>...</think>) structure for Tasks D and E.
    """
    has_think_start = "<think>" in prediction
    has_think_end = "</think>" in prediction
    
    if not has_think_start:
        return False, "Missing opening <think> tag"
    if not has_think_end:
        return False, "Missing closing </think> tag"
    
    # Verify content inside think tags
    think_content = prediction.split("<think>")[1].split("</think>")[0].strip()
    if len(think_content.split()) < 10:
        return False, "Degenerate / empty reasoning block"
        
    # Verify concluding recommendation exists after </think>
    post_think = prediction.split("</think>")[1].strip()
    if len(post_think.split()) < 5:
        return False, "Missing recommendation / synthesis block following </think>"
        
    return True, "Valid"


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class AutoregressiveBenchmark:
    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_id: str = "CamiloEmiliano/yarnball-qwen-7b-lora",
        device: str = "cuda"
    ):
        self.device = device
        print(f"[*] Initializing 4-bit NF4 Quantization config on {device}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        print(f"[*] Loading Tokenizer from {adapter_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_id)
        
        print(f"[*] Loading Base Model from {base_model_id}...")
        t0 = time.time()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print(f"[+] Base model loaded in {time.time() - t0:.1f}s")
        
        print(f"[*] Attaching LoRA Adapter from {adapter_id}...")
        t1 = time.time()
        self.model = PeftModel.from_pretrained(base_model, adapter_id)
        self.model.eval()
        print(f"[+] LoRA adapter attached in {time.time() - t1:.1f}s")
        
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"[+] Resident GPU VRAM: {(total_mem - free_mem)/1024**3:.2f} GB / {total_mem/1024**3:.2f} GB")

    def format_prompt(self, user_prompt: str) -> str:
        """Formats the raw SFT prompt using standard ChatML format."""
        system_content = (
            "You are YarnBall's Unified Financial Intelligence Engine, specialized in "
            "SEC Form 10-K/10-Q property graph extraction, 5-axis event edge extraction, "
            "openCypher text-to-query compilation, multi-hop supply chain contagion reasoning, "
            "and institutional portfolio hedging."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_single(self, prompt: str, max_new_tokens: int = 512) -> Tuple[str, float, int, float]:
        """Runs free-running autoregressive generation for a single prompt."""
        formatted_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        t0 = time.time()
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        gen_tokens = output_tokens[0][input_len:]
        num_tokens = len(gen_tokens)
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
        
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text, elapsed, num_tokens, tokens_per_sec

    def run_benchmark(
        self,
        test_file: str,
        output_results_path: str,
        output_scorecard_path: str,
        samples_per_task: int = 20
    ):
        print(f"\n[*] Loading test samples from {test_file}...")
        samples_by_task = defaultdict(list)
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                task = record.get("metadata", {}).get("task_type", "UNKNOWN")
                samples_by_task[task].append(record)
                
        print(f"[+] Found {len(samples_by_task)} tasks:")
        for t, s_list in samples_by_task.items():
            print(f"    - {t}: {len(s_list)} total samples (evaluating up to {samples_per_task})")

        eval_set = []
        for task, s_list in sorted(samples_by_task.items()):
            selected = s_list[:samples_per_task]
            eval_set.extend(selected)

        print(f"\n[*] Starting Autoregressive Evaluation across {len(eval_set)} total samples...")
        results = []
        task_metrics = defaultdict(lambda: {
            "total": 0,
            "ast_valid": 0,
            "cik_clean": 0,
            "cot_valid": 0,
            "latencies": [],
            "token_counts": [],
            "tokens_per_sec": []
        })

        t_start = time.time()
        with open(output_results_path, 'w', encoding='utf-8') as out_f:
            for idx, item in enumerate(eval_set, 1):
                sample_id = item["sample_id"]
                prompt = item["prompt"]
                gt = item["target_completion"]
                task = item.get("metadata", {}).get("task_type", "UNKNOWN")

                pred_text, latency, num_tokens, tok_sec = self.generate_single(prompt)

                # Syntax and CIK gates
                cypher_valid, cypher_err = validate_cypher_syntax(pred_text)
                cik_valid, pred_ciks, gt_ciks = validate_cik_integrity(pred_text, gt)
                cot_valid, cot_err = validate_reasoning_cot(pred_text) if task in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION") else (True, "N/A")

                # Accumulate metrics
                tm = task_metrics[task]
                tm["total"] += 1
                if cypher_valid:
                    tm["ast_valid"] += 1
                if cik_valid:
                    tm["cik_clean"] += 1
                if cot_valid:
                    tm["cot_valid"] += 1
                tm["latencies"].append(latency)
                tm["token_counts"].append(num_tokens)
                tm["tokens_per_sec"].append(tok_sec)

                eval_record = {
                    "sample_id": sample_id,
                    "task_type": task,
                    "prompt": prompt,
                    "ground_truth": gt,
                    "prediction": pred_text,
                    "latency_sec": round(latency, 3),
                    "generated_tokens": num_tokens,
                    "tokens_per_sec": round(tok_sec, 1),
                    "cypher_ast_valid": cypher_valid,
                    "cypher_error": cypher_err if not cypher_valid else None,
                    "cik_clean": cik_valid,
                    "predicted_ciks": pred_ciks,
                    "ground_truth_ciks": gt_ciks,
                    "cot_valid": cot_valid,
                    "cot_error": cot_err if not cot_valid else None
                }
                results.append(eval_record)
                out_f.write(json.dumps(eval_record) + "\n")
                out_f.flush()

                # Print progress
                status_icon = "✓" if (cypher_valid and cik_valid and cot_valid) else "✗"
                print(f"[{idx:03d}/{len(eval_set):03d}] {status_icon} {task:<25} | {num_tokens:3d} toks | {latency:5.2f}s ({tok_sec:4.1f} tok/s)")

        total_elapsed = time.time() - t_start

        # Compute summary scorecard
        scorecard = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_samples": len(eval_set),
            "total_elapsed_sec": round(total_elapsed, 2),
            "overall_tokens_per_sec": round(sum(r["tokens_per_sec"] for r in results) / len(results), 2),
            "tasks": {}
        }

        print("\n" + "=" * 75)
        print("PHASE 2 AUTOREGRESSIVE BENCHMARK SCORECARD")
        print("=" * 75)
        print(f"{'Task Name':<28} {'Count':<6} {'AST/Syntax %':<14} {'CIK Clean %':<12} {'Avg Tok':<9} {'Avg Latency'}")
        print("-" * 75)

        for task, m in sorted(task_metrics.items()):
            n = m["total"]
            ast_pct = (m["ast_valid"] / n) * 100 if n > 0 else 0
            cik_pct = (m["cik_clean"] / n) * 100 if n > 0 else 0
            cot_pct = (m["cot_valid"] / n) * 100 if n > 0 else 0
            avg_tok = sum(m["token_counts"]) / n if n > 0 else 0
            avg_lat = sum(m["latencies"]) / n if n > 0 else 0
            avg_tps = sum(m["tokens_per_sec"]) / n if n > 0 else 0

            scorecard["tasks"][task] = {
                "sample_count": n,
                "syntax_valid_percent": round(ast_pct, 2),
                "cik_clean_percent": round(cik_pct, 2),
                "cot_valid_percent": round(cot_pct, 2),
                "avg_generated_tokens": round(avg_tok, 1),
                "avg_latency_sec": round(avg_lat, 2),
                "avg_tokens_per_sec": round(avg_tps, 2)
            }

            print(f"{task:<28} {n:<6} {ast_pct:>10.1f}%    {cik_pct:>9.1f}%    {avg_tok:>7.1f}   {avg_lat:>6.2f}s ({avg_tps:>4.1f} t/s)")

        print("=" * 75)
        print(f"[+] Total benchmark finished in {total_elapsed / 60:.2f} minutes.")
        print(f"[+] Logged detailed records to: {output_results_path}")
        print(f"[+] Saved scorecard to: {output_scorecard_path}")

        with open(output_scorecard_path, 'w', encoding='utf-8') as f:
            json.dump(scorecard, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Autoregressive Benchmark Harness")
    parser.add_argument("--test-file", default="data/sft/yarnball_sft_test.jsonl")
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--output-results", default="data/sft/benchmark_autoregressive_results.jsonl")
    parser.add_argument("--output-scorecard", default="data/sft/benchmark_autoregressive_scorecard.json")
    args = parser.parse_args()

    bench = AutoregressiveBenchmark()
    bench.run_benchmark(
        test_file=args.test_file,
        output_results_path=args.output_results,
        output_scorecard_path=args.output_scorecard,
        samples_per_task=args.samples_per_task
    )


if __name__ == "__main__":
    main()
