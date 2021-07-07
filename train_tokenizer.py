from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
model_dir = "./model"  # ${MODEL_DIR}
# load dataset
# dataset = load_dataset("oscar", "unshuffled_deduplicated_bn", split="train", streaming=True)

dataset = load_dataset("mc4", "bn", split="train")
# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()
# Instantiate normalizer
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Replace("\u09e4", "\u0964"),
        normalizers.Replace("\u09e5", "\u0965"),
        normalizers.Replace("\u007c", "\u0964"),
        normalizers.Replace("\u09f7", "\u0964"),
        normalizers.Replace(Regex(r"(?<=[\u0980-\u09ff]):"), "\u0983"),
        normalizers.Lowercase(),
    ]
)
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]
# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=50265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
# print(tokenizer.tokenize("জীবনে সবচেয়ে মূল্যবান জিনিস হচ্ছে"))
# Save files to disk
tokenizer.save(f"{model_dir}/tokenizer.json")