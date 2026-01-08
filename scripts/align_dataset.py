import argparse
import json
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer

def load_tokenizer(tokenizer_type, model_name_or_path=None, vocab_file=None):
    t = tokenizer_type.lower()
    if t == "bert":
        if not vocab_file:
            raise ValueError("--tokenizer_type bert requires --vocab_file")
        return BertTokenizer.from_pretrained(vocab_file)
    elif t == "roberta":
        if not model_name_or_path:
            raise ValueError("--tokenizer_type roberta requires --model_name_or_path")
        return RobertaTokenizer.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--tokenizer_type", required=True, choices=["bert", "roberta"])
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--vocab_file", default=None)
    ap.add_argument("--max_seq_length", type=int, default=128)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_type, args.model_name_or_path, args.vocab_file)

    # How many special tokens the tokenizer adds for a single sequence
    n_special = tok.num_special_tokens_to_add(pair=False)
    if n_special < 2:
        # practically should be 2 for BERT/RoBERTa single sequences
        n_special = 2

    with open(args.input_file, "r") as f_in, open(args.output_file, "w") as f_out:
        for idx, line in enumerate(tqdm(f_in)):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            orig = data.get("orig_code") or " "
            pos  = data.get("positive_code") or " "
            neg  = data.get("negative_code") or " "

            # IMPORTANT: pad to max_length to match pretrain
            enc = tok(
                orig,
                max_length=args.max_seq_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            input_ids_len = len(enc["input_ids"])
            assert input_ids_len == args.max_seq_length

            # Real (non-pad) token count
            real_len = int(sum(enc["attention_mask"]))  # includes specials

            # Build a tree-token list that matches real_len first, then pad to max_seq_length
            # Convention: use 0 for [CLS]/<s>, code tokens, and [SEP]</s> (since you don't have real AST ids yet)
            tl = [0] * real_len

            pad_len = args.max_seq_length - len(tl)
            if pad_len < 0:
                tl = tl[:args.max_seq_length]
            else:
                if tok.padding_side == "right":
                    tl = tl + [0] * pad_len
                else:
                    tl = [0] * pad_len + tl

            assert len(tl) == args.max_seq_length

            if idx == 0:
                print("padding_side =", tok.padding_side, "pad_token_id =", tok.pad_token_id)
                print("row0: len(input_ids) =", input_ids_len, "real_len(attn) =", real_len, "len(tl) =", len(tl))
                print("orig head:", repr(orig[:100]))

            n_special = tok.num_special_tokens_to_add(pair=False)  # 2 for CodeBERT
            content_len = args.max_seq_length - n_special          # 126 when max_seq_length=128

            out = {
                "orig_code": orig,
                "positive_code": pos,
                "negative_code": neg,
                "tree_token_ids": " ".join(["0"] * content_len)
            }
            f_out.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    main()
