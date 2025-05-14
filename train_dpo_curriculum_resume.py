import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
import torch
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datasets import Dataset, concatenate_datasets
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    set_seed
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
import multiprocessing

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# 添加终端输出
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
'''
Reference Model固定不变
'''
@dataclass
class ScriptArguments:
    # Model arguments
    model_name: str = field(
        default="HuggingFaceH4/zephyr-7b-beta",
        metadata={"help": "The model to train"}
    )
    sft_adapter: str = field(
        default="alignment-handbook/zephyr-7b-sft-qlora",
        metadata={"help": "The SFT adapter to use"}
    )
    
    # Data arguments
    dataset_path: str = field(
        default="/data/duxuehong/curriculum/ultrafeedback_curriculum_dpo_pairs.json",
        metadata={"help": "Path to the training dataset"}
    )
    
    # Training arguments
    output_dir: str = field(
        default="dpo_curriculum_results",
        metadata={"help": "Output directory for checkpoints and models"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU for training"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=5e-7,
        metadata={"help": "Learning rate"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum prompt length"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save steps"}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta parameter"}
    )
    
    # Wandb arguments
    wandb_project: str = field(
        default="curriculum-dpo",
        metadata={"help": "Wandb project name"}
    )
    wandb_run_name: str = field(
        default="dpo-curriculum-learning",
        metadata={"help": "Wandb run name"}
    )

@dataclass
class ConversationExample:
    user_prompt: str
    chosen: str
    rejected: str

def apply_chat_template(dataset: Dataset, tokenizer) -> Dataset:
    """Apply chat template to the dataset."""
    def process(example):
        # Convert to the format expected by apply_chat_template
        prompt_messages = [{"role": "user", "content": example["prompt"]}]
        chosen_messages = [{"role": "assistant", "content": example["chosen"]}]
        rejected_messages = [{"role": "assistant", "content": example["rejected"]}]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False) + tokenizer.eos_token
        rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False) + tokenizer.eos_token
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "difficulty": example["difficulty"]  # Preserve difficulty label
        }
    
    # Apply processing to the dataset
    processed_dataset = dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    
    return processed_dataset

def load_single_difficulty_dataset(file_path: str, tokenizer, difficulty: str) -> Dataset:
    """Load dataset for a single difficulty level"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        conv = item[difficulty]['conversation']
        if len(conv) == 2 and conv[0]['role'] == 'user' and conv[1]['role'] == 'assistant':
            examples.append(
                ConversationExample(
                    user_prompt=conv[0]['content'],
                    chosen=conv[1]['chosen_content'],
                    rejected=conv[1]['rejected_content']
                )
            )
    
    dataset = Dataset.from_list([{
        'prompt': ex.user_prompt,
        'chosen': ex.chosen,
        'rejected': ex.rejected,
        'difficulty': difficulty
    } for ex in examples])
    
    # Apply chat template
    dataset = apply_chat_template(dataset, tokenizer)
    
    return dataset

def train_dpo_curriculum(args: ScriptArguments):
    # 确定从哪个难度开始继续训练
    difficulties = ['easy', 'medium', 'hard']
    resume_epoch = 1
    resume_difficulty = 'easy'
    
    # 检查之前的训练检查点
    for epoch, difficulty in enumerate(difficulties, 1):
        checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch}_{difficulty}")
        if os.path.exists(checkpoint_dir):
            resume_epoch = epoch + 1
            logger.info(f"Resuming training from {epoch}_{difficulty}")
            if resume_epoch <= len(difficulties):
                resume_difficulty = difficulties[epoch]
    
    # Initialize wandb，设置resume=True以继续上一次的run
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.model_name,
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "num_epochs": 3
        },
        resume=True  # 继续上一次的run
    )
    
    # Model and tokenizer initialization
    model_name = args.model_name
    sft_adapter = args.sft_adapter

    # 1.加载基本模型
    model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      device_map={"": 0}, 
      torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

    # 2.加载适配器
    # 加载使用 SFT 进行微调的适配器，将该适配器命名为“DPO”，并使其可训练（is_trainable=True）
    # 对于参考模型，使用不同的名称（例如“reference”）第二次加载适配器
    # 注意：使用可训练适配器进行双重加载会生成有关不兼容密钥的非常长的 PyTorch 警告。可以安全地忽略它。
    # 基础模型现在有两个适配器：一个用于初始化 DPO 训练并将更新，另一个用于参考

    # 首先加载SFT adapter作为reference model（保持不变）
    model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=False, adapter_name="reference")
    
    # 如果是继续训练，加载上一个epoch的DPO adapter
    if resume_epoch > 1:
        last_epoch = resume_epoch - 1
        last_difficulty = difficulties[last_epoch - 1]
        last_checkpoint_dir = os.path.join(args.output_dir, f"epoch_{last_epoch}_{last_difficulty}", "DPO")
        logger.info(f"Resuming training from checkpoint: {last_checkpoint_dir}")
        model.load_adapter(last_checkpoint_dir, adapter_name="DPO", is_trainable=True)
    else:
        # 首次训练，加载原始SFT adapter作为DPO adapter
        model.load_adapter(sft_adapter, adapter_name="DPO", is_trainable=True)

    # Set seed for reproducibility
    set_seed(42)
    
    # 3.初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # 4.Training arguments base config
    base_training_args = {
        "output_dir": args.output_dir,
        "beta": args.beta,
        "num_train_epochs": 1,  # 每个难度训练一个epoch
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "linear",
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": True,
        "remove_unused_columns": False,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "model_adapter_name": "DPO",
        "ref_adapter_name": "reference",
        "report_to": "wandb",
    }

    # 5.训练循环，从resume_epoch开始继续训练
    for epoch, difficulty in enumerate(difficulties[resume_epoch-1:], resume_epoch):
        logger.info(f"Starting epoch {epoch} with {difficulty} difficulty")
        
        # Load dataset for current difficulty
        current_dataset = load_single_difficulty_dataset(
            args.dataset_path, 
            tokenizer, 
            difficulty
        )
        
        # Update output directory for current difficulty
        current_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_{difficulty}")
        current_args = DPOConfig(**{
            **base_training_args,
            "output_dir": current_output_dir
        })

        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=model,  # 使用当前的model，它包含了之前训练的结果
            args=current_args,
            train_dataset=current_dataset,
            processing_class=tokenizer,
        )

        # Train
        train_result = dpo_trainer.train()
        
        # Log metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(current_dataset)
        metrics["difficulty"] = difficulty
        metrics["epoch"] = epoch
        dpo_trainer.log_metrics("train", metrics)
        dpo_trainer.save_metrics("train", metrics)
        
        # Save checkpoint
        dpo_trainer.save_model(current_output_dir)
        
        # 如果不是最后一个难度，需要为下一个难度准备模型
        if epoch < len(difficulties):
            # 删除当前的DPO adapter
            model.delete_adapter("DPO")
            # 加载上一轮训练好的adapter作为新的DPO adapter
            DPO_adapter_path = os.path.join(current_output_dir, "DPO")
            logger.info(f"Loading adapter from: {DPO_adapter_path}")
            model.load_adapter(DPO_adapter_path, adapter_name="DPO", is_trainable=True)
        
        logger.info(f"Completed epoch {epoch} with {difficulty} difficulty")
    
    # Save the final model
    final_output_dir = os.path.join(args.output_dir, "final_model_curriculum")
    model.save_pretrained(final_output_dir)
    
    # End wandb run
    wandb.finish()

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    logger.info(f"Starting curriculum DPO training with arguments: {args}")
    train_dpo_curriculum(args)
    logger.info("Completed curriculum DPO training")

if __name__ == "__main__":
    main() 