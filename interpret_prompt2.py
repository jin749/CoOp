import os
import sys
import argparse
import torch

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


def load_clip_to_cpu(backbone_name="ViT-B/32"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# parser = argparse.ArgumentParser()
# parser.add_argument("fpath", type=str, help="Path to the learned prompt")
# parser.add_argument("topk", type=int, help="Select top-k similar words")
# args = parser.parse_args()

# fpath = args.fpath
# topk = args.topk

fpath = "/hdd/hdd3/jsh/coop_main/output(valila..)/oxford_pets/CoOp/vit_b32_16shots/nctx16_cscTrue_ctpend/seed1/prompt_learner/model.pth.tar-200"
#fpath = "/hdd/hdd3/jsh/coop_concept44/output/output44/oxford_pets/CoOp/vit_b32_16shots/nctx16_cscTrue_ctpend/seed1/prompt_learner/model.pth.tar-200"
topk = 5

assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["ctx"]
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")
# basic_char_for_writing = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
#             "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
#             "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", ".", ",", "?", "-", ":", ";", "(", ")", "<", ">", "/", "'", '"']
basic_char_for_writing = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "!", ".", ",", "?", "-", ":", ";", "(", ")", "<", ">", "/", "'", '"']


if ctx.dim() == 2:
    # # Generic context
    # distance = torch.cdist(ctx, token_embedding)
    # print(f"Size of distance matrix: {distance.shape}")
    # sorted_idxs = torch.argsort(distance, dim=1)
    # sorted_idxs = sorted_idxs[:, :topk]

    # for m, idxs in enumerate(sorted_idxs):
    #     words = [tokenizer.decoder[idx.item()] for idx in idxs]
    #     dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
    #     print(f"{m+1}: {words} {dist}")
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)

    closest_idxs = []
    for i, idxs in enumerate(sorted_idxs):
        t2 = []
        k = 0
        for idx in idxs:
            token = tokenizer.decoder[idx.item()]
            if (set(token) - set(basic_char_for_writing)) == set():
                t2.append(idx)
                k += 1
            if k == topk:
                break
        closest_idxs.append(t2)

    #sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(closest_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m+1}: {words} {dist}")

elif ctx.dim() == 3:
    for i, c in enumerate(ctx):
        print(f"Class {i}")

        # Generic context
        distance = torch.cdist(c, token_embedding)
        print(f"Size of distance matrix: {distance.shape}")
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            print(f"{m+1}: {words} {dist}")

        # # Class-specific context
        # distance = torch.cdist(c, token_embedding)
        # print(f"Size of distance matrix: {distance.shape}")
        # sorted_idxs = torch.argsort(distance, dim=1)

        # closest_idxs = []
        # for i, idxs in enumerate(sorted_idxs):
        #     t2 = []
        #     k = 0
        #     for idx in idxs:
        #         token = tokenizer.decoder[idx.item()]
        #         if (set(token) - set(basic_char_for_writing)) == set():
        #             t2.append(idx)
        #             k += 1
        #         if k == topk:
        #             break
        #     closest_idxs.append(t2)

        # #sorted_idxs = sorted_idxs[:, :topk]

        # for m, idxs in enumerate(closest_idxs):
        #     words = [tokenizer.decoder[idx.item()] for idx in idxs]
        #     dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        #     print(f"{m+1}: {words} {dist}")

print("Done")
