from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model.tokenizer.save_pretrained("../site/tokenizer")
save_file(model[0].state_dict(), "../site/model.safetensors")
