#!/usr/bin/env python3
"""
Llama 3.3 Batch Processing Script

This script reads a JSONL file containing documents with spans, applies a sliding
window over the spans, generates a Llama 3.3 API request payload for each window,
sends the request to the local Llama model, and writes the results
(response or error) to an output JSONL file.

Requirements:
- transformers
- torch
- accelerate
- sentencepiece
"""

import os
import json
import time
import sys
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for batch processing"""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens: int = 2048
    window_size: int = 20
    step: int = 10
    temperature: float = 0.0
    top_p: float = 0.0
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    batch_size: int = 1
    max_length: int = 4096

class LlamaBatchProcessor:
    """Handles batch processing of documents using Llama 3.3"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Llama model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.device
            
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                cache_dir=os.environ.get("HF_HOME", None)
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=os.environ.get("HF_HOME", None)
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _format_prompt(self, system_instruction: str, user_content: str) -> str:
        """Format the prompt for Llama 3.3 using proper chat format"""
        # Llama 3.3 Instruct chat format
        prompt = f"""<|system|>\n{system_instruction}<|end|>\n<|user|>### Input spans:\n{user_content}### Now output the result ONLY in the example JSON format:<|end|>\n<|assistant|>\n"""
        #print(prompt)
        return prompt
    
    def _generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Llama model"""
        try:
            # Set random seed for deterministic behavior
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length - self.config.max_tokens
            ).to(self.device)
            
            # Generate response with deterministic settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=0.0,  # Set to 0 for deterministic behavior
                    do_sample=False,  # Disable sampling for deterministic output
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).split("<|end|>")[0].strip()
            #print(response_text)
            #exit()

            return {
                "content": response_text.strip(),
                "model": self.config.model_name,
                "usage": {
                    "prompt_tokens": inputs['input_ids'].shape[1],
                    "completion_tokens": len(outputs[0]) - inputs['input_ids'].shape[1],
                    "total_tokens": len(outputs[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def process_document_with_llama(
        self,
        input_path: str,
        output_path: str,
        system_instruction_path: str
    ) -> None:
        """
        Process documents with Llama 3.3 model
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            system_instruction_path: Path to system instruction file
        """
        
        # Read system instruction
        try:
            with open(system_instruction_path, 'r', encoding='utf-8') as f_sys:
                system_instruction = f_sys.read().strip()
        except FileNotFoundError:
            logger.error(f"System instruction file not found at {system_instruction_path}")
            return
        except Exception as e:
            logger.error(f"Error reading system instruction file: {e}")
            return
        
        if not system_instruction:
            logger.warning(f"System instruction file at {system_instruction_path} is empty")
        
        # Initialize counters
        processed_windows_count = 0
        failed_windows_count = 0
        total_windows = 0
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                
                for line_num, line in enumerate(infile, 1):
                    try:
                        data = json.loads(line)
                        doc = data.get('Doc')
                        spans = data.get('Spans')
                        
                        if not doc or not spans or not isinstance(spans, list):
                            logger.warning(f"Skipping malformed input line {line_num}: missing Doc or Spans")
                            continue
                        
                        num_spans = len(spans)
                        
                        # Slide window over spans
                        for start_idx in range(0, num_spans, self.config.step):
                            total_windows += 1
                            end_idx = min(start_idx + self.config.window_size, num_spans)
                            window = spans[start_idx:end_idx]
                            
                            if not window:
                                continue
                            
                            # Determine span numbers for identifier
                            try:
                                start_no = int(window[0].get('span_no', 0))
                                end_no = int(window[-1].get('span_no', 0))
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error getting span numbers for doc {doc}, window {start_idx}: {e}")
                                start_no = 0
                                end_no = 0
                            
                            # Format identifier
                            request_id = f"{doc}_spansection_{start_no}-{end_no}"
                            
                            # Convert window to JSON string
                            window_json = json.dumps(window, ensure_ascii=False)
                            
                            # Format prompt
                            prompt = self._format_prompt(system_instruction, window_json)
                            
                            # Initialize response object
                            response_obj = {"id": request_id}
                            
                            try:
                                logger.info(f"Processing window {request_id}...")
                                
                                # Generate response
                                response = self._generate_response(prompt)
                                
                                # Add response to output object
                                response_obj["response"] = response
                                
                                processed_windows_count += 1
                                logger.info(f"  → Success for {request_id}")
                                
                            except Exception as e:
                                logger.error(f"  → Error for {request_id}: {e}")
                                response_obj["error"] = str(e)
                                failed_windows_count += 1
                            
                            # Write result to output file
                            outfile.write(json.dumps(response_obj, ensure_ascii=False) + "\n")
                            outfile.flush()
                            
                            # Add small delay to prevent overwhelming the system
                            time.sleep(0.1)
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Skipping invalid JSON input line {line_num}: {e}")
                        failed_windows_count += 1
                    
                    except Exception as e:
                        logger.error(f"Unexpected error processing line {line_num}: {e}")
                        failed_windows_count += 1
        
        except FileNotFoundError:
            logger.error(f"Input data file not found at {input_path}")
            return
        except Exception as e:
            logger.error(f"Unrecoverable error during file processing: {e}")
            return
        
        # Print summary
        logger.info("\nProcessing complete.")
        logger.info(f"Total windows attempted: {total_windows}")
        logger.info(f"Successfully processed windows: {processed_windows_count}")
        logger.info(f"Failed windows: {failed_windows_count}")


def main():
    """Main function with example usage"""
    
    # Define file paths
    input_data_file = "../revised_data_explicit_and_altlex.jsonl"
    system_instruction_file = "../prompts_explicit.txt"
    output_results_file = "llama_results_explicit.jsonl"
    
    # Configuration
    config = ProcessingConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=2048,
        window_size=20,
        step=10,
        temperature=0.0,
        top_p=0.0,
        device="auto",
        max_length=4096
    )
    
    # Create processor
    processor = LlamaBatchProcessor(config)
    
    # Process documents
    processor.process_document_with_llama(
        input_path=input_data_file,
        output_path=output_results_file,
        system_instruction_path=system_instruction_file
    )
    
    logger.info(f"Results written to {output_results_file}")


if __name__ == "__main__":
    main() 