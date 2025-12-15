from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
model = AutoModel.from_pretrained("intfloat/e5-small")

def embed_chunks(chunks):
    inputs = tokenizer(
        ["passage: " + chunk for chunk in chunks],
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    normed = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
    return normed.numpy()
