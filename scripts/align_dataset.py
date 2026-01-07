import argparse
import json
import os
from transformers import RobertaTokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="downloads/huggingface/codebert-base")
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name_or_path}")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    except Exception as e:
        print(f"Failed to load from {args.model_name_or_path}: {e}")
        print("Falling back to microsoft/codebert-base")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print(f"Processing {args.input_file} -> {args.output_file}")
    
    with open(args.input_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            orig_code = data.get("orig_code", "")
            if orig_code is None: orig_code = " "
            
            # Tokenize same way as training script
            tokens = tokenizer(
                orig_code,
                max_length=args.max_seq_length,
                truncation=True,
                padding=False, # Script uses False unless pad_to_max_length is set (defaults False)
                return_special_tokens_mask=True
            )
            input_ids = tokens['input_ids']
            
            # Calculate needed tree tokens (remove cls and sep)
            # The training script does:
            # tl = [CLS] + tree_tokens[:max-2] + [SEP]
            # assert len(tl) == len(input_ids)
            # So len(tree_tokens_used) must be len(input_ids) - 2
            
            num_tree_tokens = max(0, len(input_ids) - 2)
            
            # Construct tree_token_ids string
            # Assuming 0 is the default/dummy value required
            tree_ids = ["0"] * num_tree_tokens
            data["tree_token_ids"] = " ".join(tree_ids)
            
            f_out.write(json.dumps(data) + "\n")
    
    print("Done.")

if __name__ == "__main__":
    main()
