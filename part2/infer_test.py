"""Generate test submission files directly from a saved T5 checkpoint."""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from load_data import T5Dataset, test_collate_fn
from train_t5 import DEVICE, test_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--experiment_name", default="experiment")
    args = parser.parse_args()
    if args.num_beams < 1 or args.batch_size < 1:
        parser.error("Beam count and batch size must be positive.")
    sql_path = Path("results") / f"t5_ft_{args.experiment_name}_test.sql"
    record_path = Path("records") / f"t5_ft_{args.experiment_name}_test.pkl"
    if sql_path.exists() or record_path.exists():
        parser.error("Output already exists. Choose a new experiment_name.")
    args.tokenizer = T5TokenizerFast.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(DEVICE)
    dataset = T5Dataset(
        "data", "test", task_prefix=getattr(model.config, "task_prefix", ""),
        schema_text=getattr(model.config, "schema_text", ""),
        max_input_length=getattr(model.config, "max_input_length", 128),
        tokenizer=args.tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=test_collate_fn)
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task prefix: {dataset.task_prefix!r}; beams: {args.num_beams}")
    test_inference(args, model, loader, str(sql_path), str(record_path))


if __name__ == "__main__":
    main()
