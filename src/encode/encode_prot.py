from transformers import BertModel, BertTokenizer
import re
from ..utils.load_config import load_config

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

config = load_config()

max_length = config["bert"].get("max_length", None)
truncate = False
if max_length is not None:
    truncate = True


def encode_protein(sequence):
    seq_w_space = " ".join(list(sequence))
    encoded_input = tokenizer(
        seq_w_space,
        return_tensors="pt",
        max_length=max_length,
        truncation=truncate,
        padding="max_length",
    )
    output = model(**encoded_input)
    output_hidden = output["last_hidden_state"][:, 0][0].detach().cpu().numpy()
    assert len(output_hidden) == 1024
    return output_hidden
