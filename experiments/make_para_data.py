from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

en_wiki = load_dataset('ltg/en-wiki-paraphrased', split='train')

min_words = 50
max_words = 100

en_wiki = en_wiki.filter(lambda x: (len(x["original"].split()) >= min_words) and (len(x["original"].split()) <= max_words))
en_wiki = en_wiki.filter(lambda x: (len(x["paraphrase"].split()) >= min_words) and (len(x["paraphrase"].split()) <= max_words))

input_ = en_wiki["original"]
output_ = en_wiki["paraphrase"]

model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
model = model.cuda()

score_thr = 0.9

input_embeddings_list = []
output_embeddings_list = []

input_list = []
output_list = []
score_list = []

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    T = tqdm(zip(range(len(input_)), input_, output_), total=len(input_))
    for ind, inp, out in T:
        input_embedding = model.encode([inp], normalize_embeddings=True)
        output_embedding = model.encode([out], normalize_embeddings=True)
        score = input_embedding[0] @ output_embedding[0].T

        if score > score_thr:
            input_embeddings_list.append(input_embedding[0].detach().cpu())
            output_embeddings_list.append(output_embedding[0].detach().cpu())
            input_list.append(inp)
            output_list.append(out)
            score_list.append(score.item())

        T.set_description(f"{len(input_embeddings_list)} out of {ind + 1}")

        if len(input_embeddings_list) == 10000:
            break

input_embeddings_list = torch.stack(input_embeddings_list)
output_embeddings_list = torch.stack(output_embeddings_list)

torch.save(
    {
        "input_embeddings": input_embeddings_list,
        "output_embeddings": output_embeddings_list,
        "input": input_list,
        "output": output_list,
        "score": score_list,
    },
    "para_embeddings.pth"
)
        