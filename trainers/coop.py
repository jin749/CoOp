import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import json

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        ####
        n_ctx = 8
        print("n_ctx: ", n_ctx)
        c_ctx = 8  # 일단 4개 concept 추가해서 실험해보자
        print("c_ctx: ", c_ctx)
        with open(f"concepts/{cfg.OUTPUT_DIR.split('/')[1]}.json", "r") as json_file:
            concept_dict = json.load(json_file)
        ###

        if ctx_init:
            raise Exception()

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx+c_ctx, ctx_dim, dtype=dtype)
            else:
                raise Exception()
            
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        #name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        #prompts = [f'{prompt_prefix} {name} which has X X X X {concept_dict[name][0]}, XX ' for name in classnames]
        front_prompts = [f'{prompt_prefix} {name}' for name in classnames]
        middle_prompt = "which has"
        end_prompts =  [f'@ @ {concept_dict[name][0]}, @ @ {concept_dict[name][1]}, @ @ {concept_dict[name][2]}, and @ @ {concept_dict[name][3]}.' for name in classnames]
        prompts = [f'{front_prompts[i]} {middle_prompt} {end_prompts[i]}' for i in range(len(classnames))]
        prompt_lens = [len(_tokenizer.encode(prompt)) for prompt in prompts]

        adj_locations = []
        concept_token = _tokenizer.encode("@")
        assert len(concept_token) == 1

        for i in range(len(prompts)):
            tokenized_prompt = clip.tokenize(prompts[i])[0]
            print(tokenized_prompt)
            index = 0
            locations = []
            while index < len(tokenized_prompt):
                if tokenized_prompt[index] == concept_token[0]:
                    locations.append(index)
                    index += 2
                else:
                    index += 1
            adj_locations.append(locations)
        
        self.adj_locations = adj_locations

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.embedding = embedding

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        #self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.dtype = dtype
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        #self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
    
        """
        from clip import clip
        backbone_name = "ViT-B/32"
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        model = torch.jit.load(model_path, map_location="cpu").eval()
        clip_model = clip.build_model(model.state_dict())


        """


    def forward(self):
        ctx = self.ctx
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        assert ctx.dim() == 3

        n_ctx = self.n_ctx
        #prefix = self.token_prefix
        embedding = self.embedding.to(ctx.device)
        adj_locations = self.adj_locations
        #suffix = self.token_suffix


        if self.class_token_position == "end":
            # prompts = torch.cat(
            #     [
            #         prefix,  # (n_cls, 1, dim)

            #         ctx,     # (n_cls, n_ctx, dim)
            #         suffix,  # (n_cls, *, dim)
            #     ],
            #     dim=1,
            # )
            prompts = torch.Tensor().to(ctx.device).type(self.dtype)
            for i, e in enumerate(embedding):
                sos = e[:1, :] # (1, 512)
                l0 = ctx[i][:n_ctx, :] # (16, 512)
                t0 = e[1+n_ctx: adj_locations[i][0]]
                l1 = ctx[i][n_ctx: n_ctx+2, :] # (2, 512)
                t1 = e[adj_locations[i][0]+4: adj_locations[i][1]]
                l2 = ctx[i][n_ctx+2: n_ctx+4, :] # (2, 512) 
                t2 = e[adj_locations[i][1]+4: adj_locations[i][2]]
                l3 = ctx[i][n_ctx+4: n_ctx+6, :] # (2, 512)
                t3 = e[adj_locations[i][2]+4: adj_locations[i][3]]
                l4 = ctx[i][n_ctx+6: n_ctx+8, :] # (2, 512)
                t4 = e[adj_locations[i][3]+4:]
                
                for j in adj_locations[i]:
                    prompt = torch.cat((sos, l0, t0, l1, t1, l2, t2, l3, t3, l4, t4), dim=0) # (77, 512)
                    prompt = prompt.unsqueeze(0) # (1, 77, 512)

                prompts = torch.cat((prompts, prompt), dim=0)

        # elif self.class_token_position == "middle":
        #     half_n_ctx = self.n_ctx // 2
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        #         ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,     # (1, 1, dim)
        #                 ctx_i_half1,  # (1, n_ctx//2, dim)
        #                 class_i,      # (1, name_len, dim)
        #                 ctx_i_half2,  # (1, n_ctx//2, dim)
        #                 suffix_i,     # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # elif self.class_token_position == "front":
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i = ctx[i : i + 1, :, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 class_i,   # (1, name_len, dim)
        #                 ctx_i,     # (1, n_ctx, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
