"""Evaluate a saved T5 checkpoint on dev with configurable beam search."""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from load_data import T5Dataset, normal_collate_fn
from train_t5 import DEVICE, eval_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--experiment_name", required=True)
    args = parser.parse_args()
    if args.num_beams < 1 or args.batch_size < 1:
        parser.error("Beam count and batch size must be positive.")
    sql_path = Path("results") / f"t5_ft_{args.experiment_name}_dev.sql"
    record_path = Path("records") / f"t5_ft_{args.experiment_name}_dev.pkl"
    if sql_path.exists() or record_path.exists():
        parser.error("Output already exists. Choose a new experiment_name.")
    args.tokenizer = T5TokenizerFast.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(DEVICE)
    dataset = T5Dataset("data", "dev")
    # Encode using the exact tokenizer saved with this checkpoint.
    dataset.inputs, dataset.targets = dataset.process_data("data", "dev", args.tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=normal_collate_fn)
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    loss, f1, record_em, sql_em, error_rate = eval_epoch(
        args, model, loader, "data/dev.sql", str(sql_path),
        "records/ground_truth_dev.pkl", str(record_path))
    print(f"Dev loss: {loss}, Record F1: {f1}, Record EM: {record_em}, SQL EM: {sql_em}")
    print(f"SQL execution error rate: {error_rate:.2%}")


if __name__ == "__main__":
    main()
