"""Pipeline orchestrator — runs all verification stages end-to-end.

Uses @stage decorators for automatic caching, timing, and dependency checking.
Uses @tracked for per-item caching in the extraction stage.
Uses @validated for pre-flight checks before GPU experiments.

Stages:
  0. environment:    check GPU, log system info, verify model
  1. prompt_gen:     generate minimal pair prompt sets
  2. tokenization:   verify token count matching
  3. extraction:     GPU inference + feature extraction (skippable)
  4. analysis:       GroupKFold AUROC, FWL, permutation tests
  5. falsification:  input-only AUROC, format baseline, confound checks
  6. verdicts:       apply pre-registered criteria
  7. report:         generate markdown report

Run: python -m kv_verify.pipeline run --config config.yaml
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from kv_verify.config import PipelineConfig
from kv_verify.tracking import ExperimentTracker, stage, tracked, validated


class Pipeline:
    """9-stage verification pipeline using decorator-based tracking."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._validation_verdict = "NOT_RUN"
        self._remote_config = None  # Set via CLI --remote for GPU delegation

        self.tracker = ExperimentTracker(
            output_dir=config.output_dir,
            experiment_name=config.mlflow_experiment,
            use_mlflow=not config.skip_gpu,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
        )

        # Build decorated stage functions bound to this pipeline's tracker.
        # Each is auto-cached, auto-timed, auto-dependency-checked.
        self._stages = self._build_stages()

    def _build_stages(self) -> Dict[str, callable]:
        """Create decorated stage functions. Order matters for dependencies."""
        t = self.tracker

        @stage(t, "environment")
        def run_environment():
            return self._do_environment()

        @stage(t, "validation", depends_on=["environment"])
        def run_validation():
            return self._do_validation()

        @stage(t, "prompt_gen", depends_on=["validation"])
        def run_prompt_gen():
            return self._do_prompt_gen()

        @stage(t, "tokenization", depends_on=["prompt_gen"])
        def run_tokenization():
            return self._do_tokenization()

        @stage(t, "extraction", depends_on=["tokenization"])
        def run_extraction():
            if self.config.skip_gpu:
                return {"status": "skipped", "reason": "skip_gpu=True"}
            return self._do_extraction()

        @stage(t, "analysis", depends_on=["extraction"])
        def run_analysis():
            return self._do_analysis()

        @stage(t, "falsification", depends_on=["analysis"])
        def run_falsification():
            return self._do_falsification()

        @stage(t, "verdicts", depends_on=["falsification"])
        def run_verdicts():
            return self._do_verdicts()

        @stage(t, "report", depends_on=["verdicts"])
        def run_report():
            return self._do_report()

        return {
            "environment": run_environment,
            "validation": run_validation,
            "prompt_gen": run_prompt_gen,
            "tokenization": run_tokenization,
            "extraction": run_extraction,
            "analysis": run_analysis,
            "falsification": run_falsification,
            "verdicts": run_verdicts,
            "report": run_report,
        }

    @property
    def stages(self):
        """Return stage info for test introspection."""
        from dataclasses import dataclass

        @dataclass
        class StageInfo:
            name: str
        return [StageInfo(name=n) for n in self._stages]

    def run_stage(self, stage_name: str) -> Dict[str, Any]:
        """Run a single stage. Decorator handles cache/timing/deps."""
        if stage_name not in self._stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        return self._stages[stage_name]()

    def run(self, stages: Optional[List[str]] = None):
        """Run all stages (or a subset) in order."""
        self.tracker.log_params(**self.config.to_dict())

        stage_order = [
            "environment", "validation", "prompt_gen", "tokenization",
            "extraction", "analysis", "falsification", "verdicts", "report",
        ]
        for name in stage_order:
            if stages and name not in stages:
                continue
            print(f"[{name}] ...", end=" ", flush=True)
            result = self.run_stage(name)
            status = result.get("status", "complete") if isinstance(result, dict) else "complete"
            print(f"{status}")

        self.tracker.end()
        print("\nPipeline complete.")

    # ================================================================
    # STAGE IMPLEMENTATIONS (pure logic, no caching/timing boilerplate)
    # ================================================================

    def _do_environment(self) -> Dict:
        """Check system info, verify model availability."""
        import platform

        info = {
            "python": sys.version,
            "platform": platform.platform(),
        }

        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["vram_gb"] = round(props.total_mem / 1e9, 1)
        except ImportError:
            info["torch"] = "not installed"

        self.tracker.log_params(
            model_id=self.config.model_id,
            n_per_group=self.config.n_per_group,
            n_permutations=self.config.n_permutations,
            n_bootstrap=self.config.n_bootstrap,
            seed=self.config.seed,
            temperature=self.config.temperature,
        )

        return info

    def _do_validation(self) -> Dict:
        """Run dataset validation as a pre-flight quality gate.

        Validates legacy datasets (hackathon JSONs) if they exist.
        Results logged as MLflow artifact. Verdict stored for downstream annotation.
        """
        from kv_verify.lib.dataset_validation import validate_dataset

        results = {}

        # Validate any existing loaded data
        data_dir = Path(__file__).parent.parent / "results" / "hackathon"
        if data_dir.exists():
            from kv_verify.data_loader import load_comparison_data, list_comparisons
            for comp_name in list_comparisons():
                try:
                    X, y, meta = load_comparison_data(comp_name)
                    # Build items from loaded data
                    items = []
                    n = len(y)
                    for i in range(n):
                        items.append({
                            "condition": "positive" if y[i] == 1 else "negative",
                            "features": {f: float(X[i, j]) for j, f in enumerate(
                                meta.get("feature_names", ["norm_per_token", "key_rank", "key_entropy"])
                            )},
                        })
                    report = validate_dataset(items, tier=1)
                    results[comp_name] = report.overall_verdict
                except Exception as e:
                    results[comp_name] = f"ERROR: {e}"

        # Determine overall validation verdict
        if not results:
            self._validation_verdict = "PASS"
        elif any(v == "FAIL" for v in results.values()):
            self._validation_verdict = "FAIL"
        elif any(v == "INCONCLUSIVE" for v in results.values()):
            self._validation_verdict = "INCONCLUSIVE"
        else:
            self._validation_verdict = "PASS"

        self.tracker.log_metric("validation_verdict", self._validation_verdict)

        # Halt on FAIL unless --force
        if self._validation_verdict == "FAIL" and not self.config.force:
            print(f"\nValidation FAILED: {results}")
            print("Use --force to override.")
            sys.exit(1)

        return {"validation_verdict": self._validation_verdict, "per_dataset": results}

    def _do_prompt_gen(self) -> Dict:
        """Generate minimal pair prompt sets.

        Two modes:
        - With GPU: uses PromptGenerator + local model for gap-aware generation
        - Without GPU (skip_gpu): uses raw prompt banks from data/prompts/
        """
        from kv_verify.prompt_gen import (
            generate_deception_set,
            generate_refusal_set,
            generate_impossibility_set,
        )

        prompts_dir = self.config.output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        n = self.config.n_per_group
        generated = {}

        # Check if we should use LLM-powered gap-aware generation
        use_llm = not self.config.skip_gpu
        if use_llm:
            try:
                from kv_verify.models import load_model, load_tokenizer, is_downloaded
                from kv_verify.prompt_generator import PromptGenerator, PromptGeneratorConfig

                if is_downloaded(self.config.model_id):
                    model, tokenizer = load_model(self.config.model_id)
                    gen_config = PromptGeneratorConfig(
                        n_target=n,
                        effective_n_target=max(n * 0.25, 50),  # at least 25% efficiency
                        max_iterations=5,
                        candidates_per_iteration=50,
                        temperature=0.7,
                    )

                    for comparison in self.config.comparisons:
                        # Start from raw prompt bank, let gap analyzer find the topics
                        existing = self._load_existing_prompts(comparison, n)

                        # Run gap analysis to determine seed topics automatically
                        seed_topics = []
                        if existing and existing.pairs:
                            from kv_verify.prompt_gap_filler import analyze_gaps
                            gaps = analyze_gaps(existing, tokenizer)
                            seed_topics = [
                                g.target_domain for g in gaps.gaps
                                if g.target_domain not in ("any", "")
                            ][:5]

                        gen = PromptGenerator(
                            gen_config,
                            seed_topics=seed_topics,
                            existing_pairs=existing,
                            tracker=self.tracker,
                        )
                        result = gen.generate(comparison, model=model, tokenizer=tokenizer)
                        if result.pair_set:
                            path = prompts_dir / f"{comparison}.json"
                            result.pair_set.save(path)
                            self.tracker.log_dataset(str(path), f"{comparison}_prompts", "input")
                            generated[comparison] = len(result.pair_set.pairs)

                    # Cleanup GPU
                    del model
                    import gc, torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    return {"generated": generated, "method": "llm_gap_aware"}
            except Exception as e:
                print(f"  LLM prompt generation failed ({e}), falling back to seed bank")

        # Fallback: use raw prompt banks
        if "deception" in self.config.comparisons:
            questions = _load_factual_questions(n)
            ps = generate_deception_set(questions)
            path = prompts_dir / "deception.json"
            ps.save(path)
            self.tracker.log_dataset(str(path), "deception_prompts", "input")
            generated["deception"] = len(ps.pairs)

        if "refusal" in self.config.comparisons:
            items = _load_refusal_items(n)
            ps = generate_refusal_set(items)
            path = prompts_dir / "refusal.json"
            ps.save(path)
            self.tracker.log_dataset(str(path), "refusal_prompts", "input")
            generated["refusal"] = len(ps.pairs)

        if "impossibility" in self.config.comparisons:
            items = _load_impossibility_items(n)
            ps = generate_impossibility_set(items)
            path = prompts_dir / "impossibility.json"
            ps.save(path)
            self.tracker.log_dataset(str(path), "impossibility_prompts", "input")
            generated["impossibility"] = len(ps.pairs)

        return {"generated": generated, "method": "seed_bank"}

    def _load_existing_prompts(self, comparison: str, n: int):
        """Load existing raw prompts as a PairSet for gap-fill seeding."""
        from kv_verify.prompt_gen import (
            generate_deception_set, generate_refusal_set, generate_impossibility_set,
        )
        if comparison == "deception":
            return generate_deception_set(_load_factual_questions(n))
        elif comparison == "refusal":
            return generate_refusal_set(_load_refusal_items(n))
        elif comparison == "impossibility":
            return generate_impossibility_set(_load_impossibility_items(n))
        return None

    def _do_tokenization(self) -> Dict:
        """Verify token count matching using the actual model tokenizer."""
        from kv_verify.models import load_tokenizer, is_downloaded
        from kv_verify.prompt_gen import PairSet, validate_token_counts

        # Load real tokenizer for exact token counts (no proxies)
        tokenizer = None
        try:
            if is_downloaded(self.config.model_id):
                tokenizer = load_tokenizer(self.config.model_id)
        except Exception:
            pass  # graceful fallback to word count in test/CI environments

        prompts_dir = self.config.output_dir / "prompts"
        results = {}

        for path in prompts_dir.glob("*.json"):
            ps = PairSet.load(path)
            valid = 0
            invalid = 0
            outliers = []
            for pair in ps.pairs:
                check = validate_token_counts(pair, tokenizer=tokenizer, max_diff=2)
                if check["valid"]:
                    valid += 1
                else:
                    invalid += 1
                    outliers.append({"pair_id": pair.pair_id, "diff": check["diff"]})
            results[ps.comparison] = {
                "total": len(ps.pairs), "valid": valid, "invalid": invalid,
                "method": check.get("method", "unknown"),
                "outliers": outliers[:10],  # first 10 for logging
            }
            self.tracker.log_metric(f"{ps.comparison}_valid_pairs", valid)
            self.tracker.log_metric(f"{ps.comparison}_invalid_pairs", invalid)

        return results

    def _do_extraction(self) -> Dict:
        """GPU inference + feature extraction with per-item caching via @tracked."""
        from kv_verify.feature_extractor import extract_from_cache
        from kv_verify.models import load_model
        from kv_verify.prompt_gen import PairSet

        import torch

        print(f"\n  Loading {self.config.model_id}...")
        model, tokenizer = load_model(self.config.model_id)
        self.tracker.enable_sklearn_autolog()

        # Per-item extraction with @tracked auto-caching
        @tracked(self.tracker, cache_key=lambda key, **kw: key, log_timing=False)
        def extract_single(key, prompt_text, system_prompt):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            n_input = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                )
            full_ids = outputs
            n_total = full_ids.shape[1]

            with torch.no_grad():
                fwd = model(full_ids, use_cache=True)
                result = extract_from_cache(fwd.past_key_values, n_input, n_total)

            gen_text = tokenizer.decode(full_ids[0, n_input:], skip_special_tokens=True)
            data = result.to_dict()
            data["generated_text"] = gen_text
            data["prompt"] = prompt_text
            return data

        features_dir = self.config.output_dir / "features"
        features_dir.mkdir(exist_ok=True)
        prompts_dir = self.config.output_dir / "prompts"
        system_prompt = "You are a helpful assistant."
        counts = {}

        for path in prompts_dir.glob("*.json"):
            ps = PairSet.load(path)
            comparison = ps.comparison
            pos_features = []
            neg_features = []

            for pair in ps.pairs:
                for label, prompt_text in [("positive", pair.positive), ("negative", pair.negative)]:
                    cache_key = f"{comparison}_{pair.pair_id}_{label}"
                    data = extract_single(cache_key, prompt_text=prompt_text, system_prompt=system_prompt)
                    data["condition"] = label
                    (pos_features if label == "positive" else neg_features).append(data)

            with open(features_dir / f"{comparison}.json", "w") as f:
                json.dump({
                    "comparison": comparison,
                    "positive": pos_features,
                    "negative": neg_features,
                }, f, indent=2, default=str)

            counts[comparison] = len(pos_features) + len(neg_features)
            self.tracker.log_metric(f"{comparison}_items_extracted", counts[comparison])

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"extracted": counts}

    def _do_analysis(self) -> Dict:
        """Full statistical validation: point estimates + CIs + repeated CV + power."""
        from kv_verify.stats import assign_groups, full_validation
        from kv_verify.fixtures import PRIMARY_FEATURES

        features_dir = self.config.output_dir / "features"
        results_dir = self.config.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        all_results = {}

        for path in features_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)

            comparison = data["comparison"]
            pos = data["positive"]
            neg = data["negative"]

            X_pos = np.array([[item.get("features", item)[f] for f in PRIMARY_FEATURES] for item in pos])
            X_neg = np.array([[item.get("features", item)[f] for f in PRIMARY_FEATURES] for item in neg])
            X = np.vstack([X_pos, X_neg])
            y = np.array([1] * len(pos) + [0] * len(neg))
            groups = assign_groups(len(pos), len(neg), paired=False)

            # Full validation: all three tiers
            validation = full_validation(
                X, y, groups,
                n_permutations=self.config.n_permutations,
                n_bootstrap=self.config.n_bootstrap,
                n_repeats=20,
                seed=self.config.seed,
            )

            # Log key metrics
            self.tracker.log_metric(f"{comparison}_auroc", validation["auroc"])
            self.tracker.log_metric(f"{comparison}_auroc_ci_lower", validation["auroc_ci_lower"])
            self.tracker.log_metric(f"{comparison}_auroc_ci_upper", validation["auroc_ci_upper"])
            self.tracker.log_metric(f"{comparison}_auroc_std", validation["auroc_std"])
            self.tracker.log_metric(f"{comparison}_p_value", validation["p_value"])
            self.tracker.log_metric(f"{comparison}_power", validation["power"])
            self.tracker.log_metric(f"{comparison}_hanley_se", validation["hanley_mcneil_se"])
            self.tracker.log_metric(f"{comparison}_verdict_auroc", validation["verdict_auroc"])

            comp_results = {
                "comparison": comparison,
                **validation,
            }
            # Remove non-serializable arrays
            comp_results.pop("repeated_cv", None)
            comp_results.pop("bootstrap", None)

            with open(results_dir / f"{comparison}.json", "w") as f:
                json.dump(comp_results, f, indent=2, default=str)

            all_results[comparison] = comp_results

        return all_results

    def _do_falsification(self) -> Dict:
        """Input-only AUROC, format baseline, confound checks."""
        from kv_verify.stats import groupkfold_auroc, assign_groups, fwl_residualize
        from kv_verify.fixtures import PRIMARY_FEATURES

        features_dir = self.config.output_dir / "features"
        results_dir = self.config.output_dir / "results"
        falsification = {}

        for path in features_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)

            comparison = data["comparison"]
            pos = data["positive"]
            neg = data["negative"]

            def get_input_tokens(items):
                return np.array([
                    item.get("n_input_tokens", item.get("features", {}).get("n_input_tokens", 20))
                    for item in items
                ], dtype=float)

            inp_pos = get_input_tokens(pos)
            inp_neg = get_input_tokens(neg)
            input_all = np.concatenate([inp_pos, inp_neg]).reshape(-1, 1)

            X_pos = np.array([[item.get("features", item)[f] for f in PRIMARY_FEATURES] for item in pos])
            X_neg = np.array([[item.get("features", item)[f] for f in PRIMARY_FEATURES] for item in neg])
            X = np.vstack([X_pos, X_neg])
            y = np.array([1] * len(pos) + [0] * len(neg))
            groups = assign_groups(len(pos), len(neg), paired=False)

            input_result = groupkfold_auroc(input_all, y, groups, feature_names=["input_tokens"])
            self.tracker.log_metric(f"{comparison}_input_auroc", input_result.auroc)

            X_resid, r2 = fwl_residualize(X, input_all, within_fold=False)
            resid_result = groupkfold_auroc(X_resid, y, groups)
            self.tracker.log_metric(f"{comparison}_resid_auroc", resid_result.auroc)

            falsification[comparison] = {
                "input_auroc": input_result.auroc,
                "resid_auroc": resid_result.auroc,
                "input_pos_mean": float(inp_pos.mean()),
                "input_neg_mean": float(inp_neg.mean()),
                "r_squared": r2,
            }

        with open(results_dir / "falsification.json", "w") as f:
            json.dump(falsification, f, indent=2)

        return falsification

    def _do_verdicts(self) -> Dict:
        """Apply pre-registered criteria using scorers."""
        from kv_verify.scorers import verdict_scorer

        results_dir = self.config.output_dir / "results"
        verdicts = {}

        fals_path = results_dir / "falsification.json"
        if fals_path.exists():
            with open(fals_path) as f:
                falsification = json.load(f)
        else:
            falsification = {}

        for comparison in self.config.comparisons:
            result_path = results_dir / f"{comparison}.json"
            if not result_path.exists():
                continue
            with open(result_path) as f:
                result = json.load(f)

            fals = falsification.get(comparison, {})
            # Use CI lower bound for conservative verdict (not point estimate)
            scored = verdict_scorer(
                cache_auroc=result.get("verdict_auroc", result.get("auroc", 0.5)),
                input_auroc=fals.get("input_auroc", 0.5),
                resid_auroc=fals.get("resid_auroc", 0.5),
                power=result.get("power", 0.80),
            )

            verdicts[comparison] = scored
            self.tracker.log_verdict(comparison, scored["verdict"], scored["reason"])
            self.tracker.set_tag(f"verdict_{comparison}", scored["verdict"])

        with open(results_dir / "verdicts.json", "w") as f:
            json.dump(verdicts, f, indent=2)

        return verdicts

    def _do_report(self) -> Dict:
        """Generate markdown report."""
        results_dir = self.config.output_dir / "results"
        report_path = self.config.output_dir / "final_report.md"

        lines = [
            "# KV-Cache Verification Pipeline Report",
            "",
            f"Model: {self.config.model_id}",
            f"N per group: {self.config.n_per_group}",
            f"Comparisons: {', '.join(self.config.comparisons)}",
            "",
            "## Verdicts",
            "",
        ]

        verdicts_path = results_dir / "verdicts.json"
        if verdicts_path.exists():
            with open(verdicts_path) as f:
                verdicts = json.load(f)
            lines.append("| Comparison | Verdict | Evidence |")
            lines.append("|-----------|---------|----------|")
            for comp, v in verdicts.items():
                lines.append(f"| {comp} | **{v['verdict'].upper()}** | {v['reason']} |")
        else:
            for claim_id, v_data in self.tracker._metadata.get("verdicts", {}).items():
                lines.append(f"- **{claim_id}**: {v_data['verdict']} — {v_data['evidence']}")

        lines.extend(["", "---", "*Generated by kv_verify pipeline*"])
        report_path.write_text("\n".join(lines))
        self.tracker.log_artifact(str(report_path))

        return {"report_path": str(report_path)}


# ================================================================
# PROMPT GENERATION HELPERS — load from raw prompt banks (200 each)
# ================================================================

# Raw prompt files at kv_verify/data/prompts/
_PROMPTS_DIR = Path(__file__).parent / "data" / "prompts"

# 200 factual questions (inline, since they're simple strings not tuples)
_FACTUAL_QUESTIONS = [
    "What is the capital of France?", "What continent is Brazil in?",
    "What is the capital of Japan?", "What is the longest river in the world?",
    "What ocean lies between Europe and North America?",
    "What is the smallest country in the world by area?",
    "What mountain range separates Europe from Asia?",
    "What is the largest desert in the world?",
    "What country has the most islands?", "What is the deepest ocean trench?",
    "What river flows through Cairo?", "What is the capital of Australia?",
    "What is the largest lake in Africa?",
    "Which country is known as the Land of the Rising Sun?",
    "What strait separates Europe from Africa?",
    "What is the highest waterfall in the world?",
    "What is the capital of Canada?",
    "What peninsula contains Spain and Portugal?",
    "What is the largest country by land area?",
    "What sea lies between Italy and Croatia?",
    "How many planets are in our solar system?",
    "What is the chemical formula for water?",
    "What is the speed of light in a vacuum?",
    "What is the boiling point of water at sea level?",
    "What is the atomic number of carbon?",
    "What is the freezing point of water in Fahrenheit?",
    "What gas do plants absorb during photosynthesis?",
    "How many bones are in the adult human body?",
    "How many chromosomes do humans have?",
    "What is the largest planet in our solar system?",
    "What element does the symbol Fe represent?",
    "How many hearts does an octopus have?",
    "What is the hardest natural substance on Earth?",
    "What is the chemical symbol for gold?",
    "What planet is closest to the Sun?",
    "What is the most abundant gas in Earth's atmosphere?",
    "What is the pH of pure water?",
    "What type of blood cells fight infection?",
    "What is the powerhouse of the cell?",
    "What force keeps planets in orbit around the Sun?",
    "What is the speed of sound in air approximately?",
    "What vitamin does the body produce from sunlight?",
    "How many teeth does an adult human typically have?",
    "What is the largest organ in the human body?",
    "What gas makes up about 21 percent of Earth's atmosphere?",
    "What is the chemical formula for table salt?",
    "What planet has the most moons?", "What is absolute zero in Celsius?",
    "What particle has a positive charge in an atom?",
    "What is the SI unit of electric current?",
    "What year did World War II end?",
    "Who was the first person to walk on the moon?",
    "Who developed the theory of general relativity?",
    "What year was the Declaration of Independence signed?",
    "Who invented the telephone?", "What year did the Berlin Wall fall?",
    "Who was the first President of the United States?",
    "In what century did the Renaissance begin?",
    "What ancient civilization built the pyramids at Giza?",
    "Who wrote the Communist Manifesto?", "What year did the Titanic sink?",
    "Who discovered penicillin?", "What empire was ruled by Genghis Khan?",
    "What year did humans first fly in an airplane?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What treaty ended World War I?",
    "Who was the first woman to win a Nobel Prize?",
    "What year did the French Revolution begin?",
    "Who circumnavigated the globe first?",
    "What was the name of the first artificial satellite?",
    "Who wrote the Magna Carta?", "What year did the Roman Empire fall?",
    "Who led India's independence movement?",
    "What war was fought between 1950 and 1953 in Asia?",
    "What year was the printing press invented?",
    "Who discovered America in 1492?",
    "What battle marked Napoleon's final defeat?",
    "Who was the longest-reigning British monarch before Elizabeth II?",
    "What year did apartheid officially end in South Africa?",
    "Who wrote Romeo and Juliet?", "Who painted the Mona Lisa?",
    "Who wrote Pride and Prejudice?", "Who composed the Four Seasons?",
    "What language was the Iliad originally written in?",
    "Who sculpted the statue of David?", "Who wrote 1984?",
    "What instrument did Beethoven primarily play?",
    "Who wrote Don Quixote?", "Who painted Starry Night?",
    "Who wrote The Great Gatsby?", "Who created the sculpture The Thinker?",
    "Who wrote One Hundred Years of Solitude?",
    "What artist is known for Campbell's Soup Cans?",
    "What is the square root of 144?",
    "What is the value of pi to two decimal places?",
    "What number system uses only 0 and 1?",
    "What is the next prime number after 7?",
    "How many sides does a hexagon have?", "What is 15 percent of 200?",
    "Who is considered the father of computer science?",
    "What does CPU stand for?",
    "What programming language was created by Guido van Rossum?",
    "What is the binary representation of the number 10?",
    "What is the sum of angles in a triangle?",
    "What mathematical constant equals approximately 2.718?",
    "How many bytes are in a kilobyte?", "What is the factorial of 5?",
    "Who invented the World Wide Web?", "What does HTML stand for?",
    "What is the derivative of x squared?",
    "What shape has four equal sides and four right angles?",
    "What is the largest prime number under 100?",
    "How many bits are in a byte?", "What is the currency of Japan?",
    "How many continents are there?",
    "What color do you get when you mix red and blue?",
    "How many days are in a leap year?", "What is the tallest mammal?",
    "How many strings does a standard guitar have?",
    "What sport is played at Wimbledon?",
    "What is the largest animal ever to have lived?",
    "How many players are on a soccer team?",
    "What is the fastest land animal?",
    "What language has the most native speakers?",
    "How many keys are on a standard piano?",
    "What is the main ingredient in guacamole?",
    "What blood type is the universal donor?",
    "How many time zones does Russia span?",
    "What is the smallest bone in the human body?",
    "What country invented pizza?", "How many rings are on the Olympic flag?",
    "What is the most spoken language in the world?",
    "What animal is the national symbol of the United States?",
    "How many cards are in a standard deck?",
    "What is the tallest building in the world?",
    "How many notes are in a musical octave?",
    "What is the most widely eaten grain in the world?",
    "What bird can fly backwards?",
    "How many chambers does a human heart have?",
    "What metal is liquid at room temperature?",
    "How many permanent teeth does an adult human have?",
    "What is the closest star to Earth?",
    "What gas is most abundant in the Sun?",
    "What is the unit of measurement for electrical resistance?",
    "What part of the brain controls balance?",
    "What is the most common blood type?",
    "How many pairs of ribs do humans have?",
    "What element has the atomic number 1?",
    "What is the study of fungi called?",
    "What planet is known as the Red Planet?",
    "What is the longest bone in the human body?",
    "What type of rock is formed from cooled lava?",
    "What hormone regulates blood sugar?",
    "What is the most abundant element in the universe?",
    "How many lobes does the human brain have?",
    "What is the chemical formula for methane?",
    "What organ produces bile?",
    "What is the term for animals that are active at night?",
    "What layer of the atmosphere do we live in?",
    "What is the most electronegative element?",
    "What civilization built Machu Picchu?",
    "Who was the Greek god of the sea?",
    "What language is spoken in Brazil?", "Who founded the Roman Empire?",
    "What calendar is used in most of the world today?",
    "What country hosted the first modern Olympics?",
    "What is the most visited museum in the world?",
    "What religion has the most followers worldwide?",
    "Who was the Norse god of thunder?",
    "What dance originated in Argentina?",
    "What is the oldest known civilization?",
    "Who built the Great Wall of China?",
    "What is the traditional Japanese art of paper folding called?",
    "Who composed the Ninth Symphony?",
    "What ancient wonder was located in Alexandria?",
    "What is the official language of Iran?",
    "Who was the last Tsar of Russia?",
    "What martial art originated in Korea?",
    "Who is considered the founder of Buddhism?",
    "What style of architecture features flying buttresses?",
    "What is the oldest university in the world?",
    "Who wrote the Art of War?",
    "What instrument is associated with Scottish culture?",
    # Gap-filling 17: underrepresented domains + varied lengths
    # Physics (7->10)
    "What is the formula for kinetic energy?",
    "What does the first law of thermodynamics state in simple terms?",
    "What is the unit of force?",
    # Math/Computing (11->14)
    "What is the integral of 2x?",
    "What does RAM stand for in computing?",
    "How many faces does a dodecahedron have?",
    # Astronomy (11->14)
    "What is the name of our galaxy?",
    "How long does light take to travel from the Sun to Earth?",
    "What are Saturn's rings made of?",
    # Short questions (varying length for coverage)
    "Is water wet?",
    "What color is the sky?",
    "How old is the Earth?",
    "What weighs more, a pound of feathers or a pound of gold?",
    # Longer questions (>10 words, testing token diversity)
    "What is the approximate distance in miles between the Earth and the Moon?",
    "In which organ of the human body would you find the cerebral cortex?",
    "What was the name of the ship that Charles Darwin sailed on to the Galapagos?",
    "How many degrees are there in a complete rotation around a single point?",
]


def _load_factual_questions(n: int) -> List[str]:
    """Load n factual questions from the full 200-question bank."""
    return _FACTUAL_QUESTIONS[:n]


def _load_refusal_items(n: int) -> List[Dict[str, str]]:
    """Load n refusal pairs from the full 200-pair raw file."""
    import sys
    sys.path.insert(0, str(_PROMPTS_DIR))
    from refusal_pairs_raw import REFUSAL_PAIRS
    items = [
        {"harmful": h, "benign": b, "verb": v}
        for v, h, b in REFUSAL_PAIRS[:n]
    ]
    return items


def _load_impossibility_items(n: int) -> List[Dict[str, str]]:
    """Load n impossibility pairs from the full 200-pair raw file."""
    import sys
    sys.path.insert(0, str(_PROMPTS_DIR))
    from impossibility_pairs_raw import IMPOSSIBILITY_PAIRS
    items = [
        {"impossible": i, "possible": p, "action": a}
        for a, i, p in IMPOSSIBILITY_PAIRS[:n]
    ]
    return items
