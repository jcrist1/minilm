from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

save_file(model[0].auto_model.state_dict(), "../site/model.safetensors")
