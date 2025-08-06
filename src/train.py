#!/usr/bin/env python
"""
Complete implementation of GPT-OSS training script for Jordan Humanitarian AI
"""

import os
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset, load_from_disk

class GPTOSSTrainer:
    def __init__(self, config_path):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Quantization config for efficient training
        quantization_config = None
        if self.config['model'].get('quantization', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        # Model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
            "use_cache": False,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully")
        return self.model, self.tokenizer
    
    def setup_peft(self):
        """Configure LoRA for efficient fine-tuning"""
        print("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora'].get('dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def load_dataset(self):
        """Load and prepare the training dataset"""
        print("Loading dataset...")
        
        dataset_path = self.config['data']['dataset_path']
        
        if Path(dataset_path).exists():
            # Load local dataset
            dataset = load_from_disk(dataset_path)
            print(f"Loaded local dataset from {dataset_path}")
        else:
            # Load from Hugging Face Hub
            dataset = load_dataset(
                self.config['data'].get('hub_dataset', 'HuggingFaceH4/Multilingual-Thinking'),
                split="train"
            )
            print(f"Loaded dataset from Hugging Face Hub")
        
        # Split dataset if validation is needed
        if self.config['training'].get('do_eval', True):
            split = dataset.train_test_split(test_size=0.1, seed=42)
            self.train_dataset = split['train']
            self.eval_dataset = split['test']
        else:
            self.train_dataset = dataset
            self.eval_dataset = None
        
        print(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"Validation samples: {len(self.eval_dataset)}")
        
        return self.train_dataset, self.eval_dataset
    
    def get_training_args(self):
        """Create training arguments"""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training'].get('eval_batch_size', 4),
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True),
            learning_rate=float(self.config['training']['learning_rate']),
            warmup_ratio=self.config['training'].get('warmup_ratio', 0.03),
            logging_steps=self.config['training'].get('logging_steps', 10),
            save_strategy=self.config['training'].get('save_strategy', 'epoch'),
            evaluation_strategy=self.config['training'].get('evaluation_strategy', 'epoch'),
            do_eval=self.config['training'].get('do_eval', True),
            fp16=self.config['training'].get('fp16', False) and torch.cuda.is_available(),
            bf16=self.config['training'].get('bf16', True) and torch.cuda.is_available(),
            optim=self.config['training'].get('optimizer', 'adamw_torch'),
            lr_scheduler_type=self.config['training'].get('lr_scheduler', 'cosine'),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        return training_args
    
    def train(self):
        """Main training function"""
        print("\n" + "="*50)
        print("Starting GPT-OSS Humanitarian Training")
        print("="*50)
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_peft()
        
        # Load dataset
        self.load_dataset()
        
        # Get training arguments
        training_args = self.get_training_args()
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config['training']['max_length'],
            dataset_text_field="messages",
        )
        
        # Start training
        print("\nTraining started...")
        train_result = trainer.train()
        
        # Save the model
        print("\nSaving model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Model saved to: {self.output_dir / 'final_model'}")
        print("="*50)
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Train GPT-OSS for humanitarian support")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/jordan_refugee.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with small dataset"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = GPTOSSTrainer(args.config)
    
    # Debug mode modifications
    if args.debug:
        trainer.config['training']['num_epochs'] = 1
        trainer.config['training']['logging_steps'] = 1
        print("Running in debug mode...")
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
