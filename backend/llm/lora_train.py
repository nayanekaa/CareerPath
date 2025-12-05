"""LoRA fine-tuning trainer for career path generation.

This module provides utilities to fine-tune language models using LoRA
(Low-Rank Adaptation) for improved career guidance generation.

Note: Full training requires GPU. If GPU is unavailable, this provides
a skeleton implementation for reference.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

# Optional imports - gracefully handle missing dependencies using dynamic import
TRANSFORMERS_AVAILABLE = False
try:
    import importlib
    torch = importlib.import_module('torch')
    _transformers = importlib.import_module('transformers')
    AutoTokenizer = getattr(_transformers, 'AutoTokenizer')
    AutoModelForCausalLM = getattr(_transformers, 'AutoModelForCausalLM')
    TrainingArguments = getattr(_transformers, 'TrainingArguments')
    Trainer = getattr(_transformers, 'Trainer')
    _peft = importlib.import_module('peft')
    LoraConfig = getattr(_peft, 'LoraConfig')
    get_peft_model = getattr(_peft, 'get_peft_model')
    TaskType = getattr(_peft, 'TaskType')
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    # Keep names defined to avoid NameError elsewhere
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    TrainingArguments = None  # type: ignore
    Trainer = None  # type: ignore
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    TaskType = None  # type: ignore
    print("Warning: transformers/peft not installed. LoRA training will be limited.")


class LoRATrainer:
    """Fine-tune small language models using LoRA."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        output_dir: str = "./backend/checkpoints"
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model_name: HuggingFace model identifier (gpt2, distilgpt2, etc.)
            lora_r: LoRA rank (smaller = faster/cheaper)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout in LoRA layers
            output_dir: Directory to save trained weights
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.has_gpu = torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model(self) -> bool:
        """Load base model and apply LoRA config."""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library not available. Cannot load model.")
            return False
        
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=["c_attn"]  # For GPT2; adjust for other models
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA applied. Trainable params: {self.get_trainable_params_count()}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_trainable_params_count(self) -> int:
        """Get count of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def prepare_dataset(self, qa_pairs: list, split_ratio: float = 0.9) -> tuple:
        """
        Prepare training dataset from Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A dictionaries
            split_ratio: Train/validation split ratio
        
        Returns:
            (train_texts, val_texts)
        """
        texts = [f"Q: {pair['question']}\nA: {pair['answer']}" for pair in qa_pairs]
        split_idx = int(len(texts) * split_ratio)
        return texts[:split_idx], texts[split_idx:]
    
    def train(
        self,
        train_texts: list,
        val_texts: list,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-4
    ) -> Optional[str]:
        """
        Fine-tune model on training data.
        
        Args:
            train_texts: Training texts
            val_texts: Validation texts
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        
        Returns:
            Path to saved model, or None if training failed
        """
        if not TRANSFORMERS_AVAILABLE:
            print("Error: Cannot train without transformers library")
            return None
        
        if self.model is None:
            if not self.load_model():
                return None
        
        if not self.has_gpu:
            print("Warning: No GPU available. Training will be slow. Consider using Google Colab.")
        
        try:
            # Create simple dataset
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.texts = texts
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    input_ids = encoding["input_ids"].squeeze()
                    attention_mask = encoding["attention_mask"].squeeze()
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": input_ids.clone()
                    }
            
            train_dataset = SimpleDataset(train_texts, self.tokenizer)
            val_dataset = SimpleDataset(val_texts, self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                save_strategy="epoch",
                eval_strategy="epoch",
                load_best_model_at_end=True,
                logging_steps=10,
                log_level="info"
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer
            )
            
            # Train
            print("Starting training...")
            trainer.train()
            
            # Save
            save_path = os.path.join(self.output_dir, "final_model")
            # Only save if model/tokenizer loaded
            if self.model is not None:
                try:
                    self.model.save_pretrained(save_path)
                except Exception as _:
                    print("Warning: failed to save model via save_pretrained")
            if self.tokenizer is not None:
                try:
                    self.tokenizer.save_pretrained(save_path)
                except Exception as _:
                    print("Warning: failed to save tokenizer via save_pretrained")
            print(f"Model saved to {save_path}")
            
            return save_path
        
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def generate(self, prompt: str, max_length: int = 100) -> Optional[str]:
        """Generate text using the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            print("Error: Model not loaded")
            return None
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.has_gpu:
                inputs = inputs.to("cuda")
                self.model.to("cuda")
            
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            print(f"Generation error: {e}")
            return None
    
    def load_pretrained(self, checkpoint_dir: str) -> bool:
        """Load a previously trained checkpoint."""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            print(f"Loaded checkpoint from {checkpoint_dir}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False


def estimate_training_time(
    num_samples: int = 50,
    seq_length: int = 512,
    model_size: str = "small",
    has_gpu: bool = True
)-> Dict[str, Any]:
    """
    Estimate training time based on dataset and hardware.
    
    Args:
        num_samples: Number of training samples
        seq_length: Sequence length
        model_size: Model size (small, medium, large)
        has_gpu: Whether GPU is available
    
    Returns:
        Dictionary with estimated times (rough estimates)
    """
    # Rough estimates (highly dependent on hardware)
    tokens_per_sample = seq_length
    total_tokens = num_samples * tokens_per_sample
    
    # Tokens per second (rough estimates)
    if has_gpu:
        tps = {"small": 5000, "medium": 2000, "large": 500}.get(model_size, 2000)
    else:
        tps = {"small": 100, "medium": 50, "large": 20}.get(model_size, 50)
    
    base_time_hours = total_tokens / (tps * 3600)
    
    # For 3 epochs
    return {
        "per_epoch_hours": base_time_hours,
        "total_3_epochs_hours": base_time_hours * 3,
        "note": "Estimates are rough; actual times depend on batch size and hardware"
    }


if __name__ == "__main__":
    # Example usage
    print("LoRA Training Module - CareerPath AI")
    print("-" * 40)
    
    if not TRANSFORMERS_AVAILABLE:
        print("transformers/peft not installed. Install with:")
        print("  pip install transformers peft torch")
    else:
        # Example: Load dataset and train
        from lora_dataset import generate_qa_pairs
        
        qa_pairs = generate_qa_pairs(n=20)  # Small dataset for demo
        
        trainer = LoRATrainer(model_name="distilgpt2")
        trainer.load_model()
        
        train_texts, val_texts = trainer.prepare_dataset(qa_pairs)
        print(f"Training on {len(train_texts)} samples, validating on {len(val_texts)}")
        
        # Check estimated time
        estimates = estimate_training_time(
            num_samples=len(train_texts),
            model_size="small",
            has_gpu=trainer.has_gpu
        )
        print(f"Estimated training time: {estimates['total_3_epochs_hours']:.2f} hours")
        
        # Optionally train (comment out for demo)
        # trainer.train(train_texts, val_texts, num_epochs=1, batch_size=4)
