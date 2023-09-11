from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model.encode(sentences)
model.tokenizer.save_pretrained("models/tokenizer")
module_dict = model.state_dict()
save_file(module_dict, "models/minilm.safetensors")
