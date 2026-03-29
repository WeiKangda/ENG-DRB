#!/usr/bin/env python3
"""
Llama Model Benchmarking Pipeline
Combines LLM processing, postprocessing, and evaluation into a single script.
"""

import json
import math
import os
import sys
import time
import argparse
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

# Hugging Face imports for Llama
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from torch.nn import functional as F
except ImportError:
    print("Warning: transformers and torch not installed. Install with: pip install transformers torch")
    print("This script requires these packages for Llama model inference.")

@dataclass
class Config:
    """Configuration for the benchmarking pipeline."""
    # Model settings
    #model_name: str = "meta-llama/Llama-3.3-70B-Instruct"  # Change to your preferred Llama model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Processing settings
    window_size: int = 20
    step: int = 10
    max_tokens: int = 3000
    
    # File paths
    input_data_path: str = "../revised_data.jsonl"
    explicit_data_path: str = "../revised_data_explicit_and_altlex.jsonl"
    implicit_data_path: str = "../revised_data_implicit.jsonl"
    system_prompt_path: str = "system_prompt.txt"
    explicit_prompt_path: str = "../prompts_explicit.txt"
    implicit_prompt_path: str = "../prompts_implicit.txt"
    output_dir: str = "results"
    
    # Evaluation settings
    partial_agreement_threshold: float = 0.5
    gold_data_path: str = "../revised_data.jsonl"

class LlamaProcessor:
    """Handles Llama model inference for document processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Llama model and tokenizer."""
        try:
            print(f"Loading model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, cache_dir=os.environ.get("HF_HOME", None))
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None,
                cache_dir=os.environ.get("HF_HOME", None)
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Model loaded successfully on {self.config.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have the correct model name and sufficient permissions.")
            sys.exit(1)
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from Llama model."""
        try:
            # Format prompt for Llama chat
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def process_document(self, doc: str, spans: List[Dict], prompt_path: Optional[str] = None) -> List[Dict]:
        """Process a document using sliding window approach."""
        results = []
        num_spans = len(spans)
        
        # Read system prompt
        prompt_file = prompt_path if prompt_path is not None else self.config.system_prompt_path
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
        except FileNotFoundError:
            system_prompt = "You are an assistant that analyzes text spans for discourse relations."
        
        # Slide window over spans
        for start_idx in range(0, num_spans, self.config.step):
            end_idx = min(start_idx + self.config.window_size, num_spans)
            window = spans[start_idx:end_idx]
            
            if not window:
                continue
            
            # Get span numbers for identifier
            try:
                start_no = int(window[0].get('span_no', 0))
                end_no = int(window[-1].get('span_no', 0))
            except (ValueError, TypeError):
                start_no = 0
                end_no = 0
            
            request_id = f"{doc}_spansection_{start_no}-{end_no}"
            
            # Create prompt
            window_json = json.dumps(window, ensure_ascii=False)
            prompt = f"{system_prompt}\n\n{window_json}"
            print(prompt)
            print("#"*30)
            #exit()
            
            # Generate response
            print(f"Processing window {request_id}...")
            response = self.generate_response(prompt)
            print(response)
            exit()
            # Store result
            result = {
                "id": request_id,
                "response": response,
                "window": window
            }
            results.append(result)
        
        return results

class PostProcessor:
    """Handles postprocessing of LLM outputs."""
    
    @staticmethod
    def check_overlap(start1, end1, start2, end2) -> bool:
        """Check if two ranges overlap."""
        try:
            start1, end1, start2, end2 = float(start1), float(end1), float(start2), float(end2)
        except (ValueError, TypeError):
            return False
        
        if start1 > end1 or start2 > end2:
            return False
        
        return max(start1, start2) <= min(end1, end2)
    
    @staticmethod
    def are_partially_agreed(sense1: Dict, sense2: Dict) -> bool:
        """Check if two sense objects meet partial agreement criteria."""
        required_keys = ["sense", "Arg1_start", "Arg1_end", "Arg2_start", "Arg2_end"]
        for key in required_keys:
            if key not in sense1 or key not in sense2:
                return False
        
        if sense1["sense"] != sense2["sense"]:
            return False
        
        arg1_overlap = PostProcessor.check_overlap(
            sense1["Arg1_start"], sense1["Arg1_end"],
            sense2["Arg1_start"], sense2["Arg1_end"]
        )
        
        arg2_overlap = PostProcessor.check_overlap(
            sense1["Arg2_start"], sense1["Arg2_end"],
            sense2["Arg2_start"], sense2["Arg2_end"]
        )
        
        return arg1_overlap and arg2_overlap
    
    @staticmethod
    def merge_sense_objects(sense1: Dict, sense2: Dict) -> Dict:
        """Merge two partially agreed sense objects."""
        required_keys = ["sense", "Arg1_start", "Arg1_end", "Arg2_start", "Arg2_end", "explicit", "confidence"]
        for key in required_keys:
            if key not in sense1 or key not in sense2:
                return sense1
        
        try:
            arg1_start1, arg1_end1 = float(sense1["Arg1_start"]), float(sense1["Arg1_end"])
            arg2_start1, arg2_end1 = float(sense1["Arg2_start"]), float(sense1["Arg2_end"])
            arg1_start2, arg1_end2 = float(sense2["Arg1_start"]), float(sense2["Arg1_end"])
            arg2_start2, arg2_end2 = float(sense2["Arg2_start"]), float(sense2["Arg2_end"])
        except (ValueError, TypeError):
            return sense1
        
        merged_sense = {
            "sense": sense1["sense"],
            "Arg1_start": min(arg1_start1, arg1_start2),
            "Arg1_end": max(arg1_end1, arg1_end2),
            "Arg2_start": min(arg2_start1, arg2_start2),
            "Arg2_end": max(arg2_end1, arg2_end2),
        }
        
        # Merge explicit values
        explicit1 = sense1.get("explicit", "")
        explicit2 = sense2.get("explicit", "")
        
        if explicit1 == explicit2:
            merged_sense["explicit"] = explicit1
        elif explicit1 == "implicit":
            merged_sense["explicit"] = explicit2
        elif explicit2 == "implicit":
            merged_sense["explicit"] = explicit1
        else:
            explicits1_parts = explicit1.split(' | ') if explicit1 else []
            explicits2_parts = explicit2.split(' | ') if explicit2 else []
            combined_explicits = sorted(list(set(explicits1_parts + explicits2_parts)))
            merged_sense["explicit"] = " | ".join(combined_explicits)
        
        # Merge confidence
        try:
            conf1 = float(sense1.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf1 = 0.0
        try:
            conf2 = float(sense2.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf2 = 0.0
        
        merged_sense["confidence"] = max(conf1, conf2)
        
        return merged_sense
    
    @staticmethod
    def process_line(json_obj: Dict) -> Dict:
        """Process a single JSON object to remove duplicates and merge senses."""
        senses = json_obj.get("Senses", [])
        
        # Remove exact duplicates
        seen_senses_str = set()
        unique_senses = []
        for sense in senses:
            try:
                sense_str = json.dumps(sense, sort_keys=True)
                if sense_str not in seen_senses_str:
                    seen_senses_str.add(sense_str)
                    unique_senses.append(sense)
            except TypeError:
                continue
        
        # Merge partial agreed senses
        current_senses = unique_senses
        merged_occurred = True
        
        while merged_occurred:
            merged_occurred = False
            next_senses_list = []
            used_indices = set()
            
            for i in range(len(current_senses)):
                if i in used_indices:
                    continue
                
                base_sense = current_senses[i]
                current_merged_sense = base_sense
                used_indices.add(i)
                
                for j in range(i + 1, len(current_senses)):
                    if j in used_indices:
                        continue
                    
                    if PostProcessor.are_partially_agreed(current_merged_sense, current_senses[j]):
                        current_merged_sense = PostProcessor.merge_sense_objects(current_merged_sense, current_senses[j])
                        used_indices.add(j)
                        merged_occurred = True
                
                next_senses_list.append(current_merged_sense)
            
            current_senses = next_senses_list
        
        # Handle same-range different-sense conflicts
        grouped_by_range = {}
        for sense in current_senses:
            try:
                arg1_start = float(sense.get("Arg1_start", math.nan))
                arg1_end = float(sense.get("Arg1_end", math.nan))
                arg2_start = float(sense.get("Arg2_start", math.nan))
                arg2_end = float(sense.get("Arg2_end", math.nan))
                range_key = f"{arg1_start}-{arg1_end}_{arg2_start}-{arg2_end}"
            except (ValueError, TypeError):
                range_key = "invalid_range"
            
            if range_key not in grouped_by_range:
                grouped_by_range[range_key] = []
            grouped_by_range[range_key].append(sense)
        
        final_senses = []
        for range_key, senses_list in grouped_by_range.items():
            if range_key == "invalid_range":
                final_senses.extend(senses_list)
                continue
            
            try:
                sorted_senses = sorted(senses_list, key=lambda s: float(s.get("confidence", -1.0)), reverse=True)
                if sorted_senses:
                    final_senses.append(sorted_senses[0])
            except Exception:
                final_senses.extend(senses_list)
        
        return {"id": json_obj.get("id"), "Senses": final_senses}
    
    @staticmethod
    def process_jsonl_file(input_filepath: str, output_filepath: str):
        """Process a JSONL file to remove duplicates and merge senses."""
        print(f"Processing started for input file: {input_filepath}")
        processed_count = 0
        error_count = 0
        
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json_obj = json.loads(line)
                    processed_obj = PostProcessor.process_line(json_obj)
                    outfile.write(json.dumps(processed_obj) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_num}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    error_count += 1
        
        print(f"Processing finished. Successfully processed {processed_count} lines.")
        if error_count > 0:
            print(f"Encountered errors on {error_count} lines.")

class Evaluator:
    """Handles evaluation of model predictions against gold standard."""
    
    @staticmethod
    def normalize_sense(sense_str: str) -> str:
        """Normalize a sense string to the second level."""
        parts = sense_str.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:2])
        return sense_str
    
    @staticmethod
    def get_covered_span_nos(start, end, all_span_nos: List) -> Set:
        """Get span numbers covered by a span [start, end]."""
        if start is None or end is None or not all_span_nos:
            return set()
        
        sorted_span_nos = sorted(all_span_nos)
        covered = set()
        
        for span_no in sorted_span_nos:
            if start <= span_no <= end:
                covered.add(span_no)
            elif span_no > end:
                break
        
        return covered
    
    @staticmethod
    def calculate_partial_agreement(gold_sense: Dict, pred_sense: Dict, all_span_nos: List) -> float:
        """Calculate partial agreement score between gold and predicted sense."""
        gold_arg1_span_nos = set()
        gold_arg2_span_nos = set()
        pred_arg1_span_nos = set()
        pred_arg2_span_nos = set()
        
        if 'Arg1_start' in gold_sense and 'Arg1_end' in gold_sense:
            gold_arg1_span_nos = Evaluator.get_covered_span_nos(gold_sense['Arg1_start'], gold_sense['Arg1_end'], all_span_nos)
        if 'Arg2_start' in gold_sense and 'Arg2_end' in gold_sense:
            gold_arg2_span_nos = Evaluator.get_covered_span_nos(gold_sense['Arg2_start'], gold_sense['Arg2_end'], all_span_nos)
        
        if 'Arg1_start' in pred_sense and 'Arg1_end' in pred_sense:
            pred_arg1_span_nos = Evaluator.get_covered_span_nos(pred_sense['Arg1_start'], pred_sense['Arg1_end'], all_span_nos)
        if 'Arg2_start' in pred_sense and 'Arg2_end' in pred_sense:
            pred_arg2_span_nos = Evaluator.get_covered_span_nos(pred_sense['Arg2_start'], pred_sense['Arg2_end'], all_span_nos)
        
        inter_arg1 = gold_arg1_span_nos.intersection(pred_arg1_span_nos)
        inter_arg2 = gold_arg2_span_nos.intersection(pred_arg2_span_nos)
        
        union_all_args = gold_arg1_span_nos.union(gold_arg2_span_nos).union(pred_arg1_span_nos).union(pred_arg2_span_nos)
        
        numerator = len(inter_arg1) + len(inter_arg2)
        denominator = len(union_all_args)
        
        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator
    
    @staticmethod
    def load_data_and_spans(path: str, id_field: str = "id", senses_field: str = "Senses", spans_field: str = "Spans") -> Tuple[Dict, Dict]:
        """Load a JSONL file, extracting senses and span numbers."""
        senses_map = {}
        span_nos_map = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                key = obj.get(id_field)
                if key is None:
                    continue
                
                senses_list = obj.get(senses_field, [])
                processed_senses = []
                for sense in senses_list:
                    if isinstance(sense, dict):
                        sense.pop('confidence', None)
                        sense['sense'] = Evaluator.normalize_sense(sense.get('sense', ''))
                        processed_senses.append(sense)
                
                senses_map[key] = processed_senses
                
                spans_list = obj.get(spans_field, [])
                span_nos = [span.get('span_no') for span in spans_list if isinstance(span, dict) and span.get('span_no') is not None]
                span_nos_map[key] = sorted(list(set(span_nos)))
        
        return senses_map, span_nos_map
    
    @staticmethod
    def compute_scores(gold_senses_map: Dict, pred_senses_map: Dict, span_nos_map: Dict, 
                      use_partial_agreement: bool = True, partial_agreement_threshold: float = 0.5) -> Dict:
        """Compute scores per item and overall."""
        per_item_scores = {}
        all_keys = sorted(list(set(gold_senses_map) | set(pred_senses_map)))
        
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_gold_count = 0
        total_pred_count = 0
        
        for key in all_keys:
            gold_senses = gold_senses_map.get(key, [])
            pred_senses = pred_senses_map.get(key, [])
            all_span_nos = span_nos_map.get(key, [])
            
            num_gold = len(gold_senses)
            num_pred = len(pred_senses)
            
            item_tp = 0.0
            item_fp = 0.0
            item_fn = 0.0
            
            total_gold_count += num_gold
            total_pred_count += num_pred
            
            if num_gold == 0 and num_pred == 0:
                per_item_scores[key] = {
                    "precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "tp": 0.0, "fp": 0.0, "fn": 0.0, "num_gold": 0, "num_pred": 0
                }
                continue
            
            if use_partial_agreement:
                if num_gold > 0 and num_pred > 0:
                    cost_matrix = np.full((num_gold, num_pred), 1.0)
                    for i, gold_s in enumerate(gold_senses):
                        for j, pred_s in enumerate(pred_senses):
                            if gold_s.get('sense') == pred_s.get('sense'):
                                pa_score = Evaluator.calculate_partial_agreement(gold_s, pred_s, all_span_nos)
                                cost_matrix[i, j] = -pa_score
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    item_partial_tp_sum = 0.0
                    matched_gold_indices = set()
                    matched_pred_indices = set()
                    
                    for r, c in zip(row_ind, col_ind):
                        pa_score = -cost_matrix[r, c]
                        if gold_senses[r].get('sense') == pred_senses[c].get('sense'):
                            item_partial_tp_sum += pa_score
                            matched_gold_indices.add(r)
                            matched_pred_indices.add(c)
                    
                    item_tp = item_partial_tp_sum
                    item_fp = num_pred - item_tp
                    item_fn = num_gold - item_tp
                
                elif num_gold > 0 and num_pred == 0:
                    item_tp = 0.0
                    item_fp = 0.0
                    item_fn = num_gold
                elif num_gold == 0 and num_pred > 0:
                    item_tp = 0.0
                    item_fp = num_pred
                    item_fn = 0.0
            
            else:
                # Exact match logic
                gold_sense_set = set()
                for gold_s in gold_senses:
                    hashable_gold_s = frozenset(gold_s.items())
                    gold_sense_set.add(hashable_gold_s)
                
                pred_sense_set = set()
                for pred_s in pred_senses:
                    hashable_pred_s = frozenset(pred_s.items())
                    pred_sense_set.add(hashable_pred_s)
                
                item_tp = len(gold_sense_set.intersection(pred_sense_set))
                item_fp = len(pred_sense_set - gold_sense_set)
                item_fn = len(gold_sense_set - pred_sense_set)
            
            item_prec = item_tp / (item_tp + item_fp) if (item_tp + item_fp) > 0 else 0.0
            item_rec = item_tp / (item_tp + item_fn) if (item_tp + item_fn) > 0 else 0.0
            item_f1 = (2 * item_prec * item_rec / (item_prec + item_rec)) if (item_prec + item_rec) > 0 else 0.0
            
            per_item_scores[key] = {
                "precision": round(item_prec, 4),
                "recall": round(item_rec, 4),
                "f1": round(item_f1, 4),
                "tp": round(item_tp, 4),
                "fp": round(item_fp, 4),
                "fn": round(item_fn, 4),
                "num_gold": num_gold,
                "num_pred": num_pred,
            }
            
            total_tp += item_tp
            total_fp += item_fp
            total_fn += item_fn
        
        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0
        
        overall_scores = {
            "precision": round(overall_prec, 4),
            "recall": round(overall_rec, 4),
            "f1": round(overall_f1, 4),
            "total_tp": round(total_tp, 4),
            "total_fp": round(total_fp, 4),
            "total_fn": round(total_fn, 4),
            "total_gold": total_gold_count,
            "total_pred": total_pred_count,
        }
        
        return {"per_item_scores": per_item_scores, "overall_scores": overall_scores}

class LlamaBenchmarkingPipeline:
    """Main pipeline class that orchestrates the entire benchmarking process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llama_processor = LlamaProcessor(config)
        self.post_processor = PostProcessor()
        self.evaluator = Evaluator()
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
    
    def run_pipeline(self, experiment_type: str = "system"):
        """Run the complete benchmarking pipeline."""
        print(f"Starting Llama Benchmarking Pipeline for {experiment_type} experiment...")
        
        # Step 1: Process documents with Llama
        print(f"\n=== Step 1: Processing {experiment_type} documents with Llama ===")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Select input data path based on experiment type
        exp_type = experiment_type.lower()
        if exp_type == "explicit":
            input_data_path = self.config.explicit_data_path
            output_filename = "llama_explicit_output.jsonl"
        elif exp_type == "implicit":
            input_data_path = self.config.implicit_data_path
            output_filename = "llama_implicit_output.jsonl"
        else:  # system
            input_data_path = self.config.input_data_path
            output_filename = "llama_raw_output.jsonl"
        
        raw_output_path = os.path.join(self.config.output_dir, output_filename)
        self._process_documents(input_data_path, raw_output_path, experiment_type=experiment_type)
        
        # Step 2: Postprocess results
        print(f"\n=== Step 2: Postprocessing {experiment_type} results ===")
        processed_output_path = os.path.join(self.config.output_dir, f"llama_processed_{exp_type}_output.jsonl")
        self.post_processor.process_jsonl_file(raw_output_path, processed_output_path)
        
        # Step 3: Evaluate results
        print(f"\n=== Step 3: Evaluating {experiment_type} results ===")
        self._evaluate_results(processed_output_path, experiment_type=experiment_type)
        
        print(f"\n{experiment_type} pipeline completed successfully!")
    
    def _process_documents(self, input_path: str, output_path: str, experiment_type: str = ""):
        """Process all documents with Llama model."""
        print(f"Processing {experiment_type} documents from: {input_path}")
        print(f"Output will be saved to: {output_path}")
        
        processed_count = 0
        error_count = 0
        
        # Select prompt path based on experiment type (case-insensitive)
        exp_type = experiment_type.lower()
        if exp_type == "explicit":
            prompt_path = self.config.explicit_prompt_path
        elif exp_type == "implicit":
            prompt_path = self.config.implicit_prompt_path
        else:
            prompt_path = self.config.system_prompt_path
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    doc = data.get('Doc')
                    spans = data.get('Spans')
                    
                    if not doc or not spans or not isinstance(spans, list):
                        print(f"Skipping malformed input line for doc: {doc}")
                        error_count += 1
                        continue
                    
                    # Process document with Llama
                    results = self.llama_processor.process_document(doc, spans, prompt_path=prompt_path)
                    
                    # Write results
                    for result in results:
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    processed_count += 1
                    print(f"Processed {experiment_type} document: {doc} ({len(results)} windows)")
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"Error processing {experiment_type} document: {e}")
                    error_count += 1
        
        print(f"{experiment_type} document processing completed. Processed: {processed_count}, Errors: {error_count}")
    
    def _evaluate_results(self, pred_path: str, experiment_type: str = ""):
        """Evaluate model predictions against gold standard."""
        print(f"Evaluating {experiment_type} predictions from: {pred_path}")
        
        # Determine gold data path based on experiment type (case-insensitive)
        exp_type = experiment_type.lower()
        if exp_type == "explicit":
            gold_data_path = self.config.explicit_data_path
        elif exp_type == "implicit":
            gold_data_path = self.config.implicit_data_path
        else:
            gold_data_path = self.config.gold_data_path
            
        print(f"Gold standard from: {gold_data_path}")
        
        # Load data
        gold_senses_map, gold_span_nos_map = self.evaluator.load_data_and_spans(
            gold_data_path, id_field="Doc", senses_field="Senses", spans_field="Spans"
        )
        
        pred_senses_map, _ = self.evaluator.load_data_and_spans(
            pred_path, id_field="id", senses_field="Senses"
        )
        
        # Filter predictions to only include documents in gold standard
        eval_doc_ids = sorted(list(gold_senses_map.keys()))
        pred_senses_map_filtered = {k: pred_senses_map.get(k, []) for k in eval_doc_ids}
        gold_span_nos_map_filtered = {k: gold_span_nos_map.get(k, []) for k in eval_doc_ids}
        
        # Compute scores with partial agreement
        print(f"\n--- Computing Scores with Partial Agreement ({experiment_type}) ---")
        partial_agreement_results = self.evaluator.compute_scores(
            {k: gold_senses_map[k] for k in eval_doc_ids},
            pred_senses_map_filtered,
            gold_span_nos_map_filtered,
            use_partial_agreement=True
        )
        
        # Print overall scores
        overall_pa = partial_agreement_results['overall_scores']
        print(f"\nOverall Scores (Partial Agreement - {experiment_type}):")
        print(f"Precision: {overall_pa['precision']}, Recall: {overall_pa['recall']}, F1: {overall_pa['f1']}")
        print(f"Total TP: {overall_pa['total_tp']}, Total FP: {overall_pa['total_fp']}, Total FN: {overall_pa['total_fn']}")
        print(f"Total Gold Senses: {overall_pa['total_gold']}, Total Predicted Senses: {overall_pa['total_pred']}")
        
        # Compute scores with exact match
        print(f"\n--- Computing Scores with Exact Match ({experiment_type}) ---")
        exact_match_results = self.evaluator.compute_scores(
            {k: gold_senses_map[k] for k in eval_doc_ids},
            pred_senses_map_filtered,
            gold_span_nos_map_filtered,
            use_partial_agreement=False
        )
        
        overall_exact = exact_match_results['overall_scores']
        print(f"\nOverall Scores (Exact Match - {experiment_type}):")
        print(f"Precision: {overall_exact['precision']}, Recall: {overall_exact['recall']}, F1: {overall_exact['f1']}")
        print(f"Total TP: {overall_exact['total_tp']}, Total FP: {overall_exact['total_fp']}, Total FN: {overall_exact['total_fn']}")
        print(f"Total Gold Senses: {overall_exact['total_gold']}, Total Predicted Senses: {overall_exact['total_pred']}")
        
        # Save results
        results_path = os.path.join(self.config.output_dir, f"evaluation_results_{exp_type}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "partial_agreement": partial_agreement_results,
                "exact_match": exact_match_results
            }, f, indent=2)
        
        print(f"\n{experiment_type} evaluation results saved to: {results_path}")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Llama Model Benchmarking Pipeline")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model name")
    parser.add_argument("--input", default="../revised_data.jsonl", help="Input data file path")
    parser.add_argument("--explicit-data", default="../revised_data_explicit_and_altlex.jsonl", help="Explicit data file path")
    parser.add_argument("--implicit-data", default="../revised_data_implicit.jsonl", help="Implicit data file path")
    parser.add_argument("--gold", default="../revised_data.jsonl", help="Gold standard data file path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size")
    parser.add_argument("--step", type=int, default=10, help="Sliding window step")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Run pipeline for explicit data
    explicit_output_dir = os.path.join(args.output_dir, "explicit")
    os.makedirs(explicit_output_dir, exist_ok=True)
    explicit_config = Config(
        model_name=args.model,
        input_data_path=args.explicit_data,
        explicit_data_path=args.explicit_data,
        implicit_data_path=args.implicit_data,
        gold_data_path=args.explicit_data,  # Use explicit data as gold for explicit run
        output_dir=explicit_output_dir,
        window_size=args.window_size,
        step=args.step,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    print("\n================ EXPLICIT DATA PIPELINE ================")
    explicit_pipeline = LlamaBenchmarkingPipeline(explicit_config)
    explicit_pipeline.run_pipeline("explicit")

    # Run pipeline for implicit data
    implicit_output_dir = os.path.join(args.output_dir, "implicit")
    os.makedirs(implicit_output_dir, exist_ok=True)
    implicit_config = Config(
        model_name=args.model,
        input_data_path=args.implicit_data,
        explicit_data_path=args.explicit_data,
        implicit_data_path=args.implicit_data,
        gold_data_path=args.implicit_data,  # Use implicit data as gold for implicit run
        output_dir=implicit_output_dir,
        window_size=args.window_size,
        step=args.step,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    print("\n================ IMPLICIT DATA PIPELINE ================")
    implicit_pipeline = LlamaBenchmarkingPipeline(implicit_config)
    implicit_pipeline.run_pipeline("implicit")

if __name__ == "__main__":
    main() 