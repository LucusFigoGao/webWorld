import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip_official

from attack import *

import os
import pickle
from tqdm import tqdm

_tokenizer = _Tokenizer()

class CustomKLDivLoss(nn.Module):
    """
    自定义的KL散度损失模块，将 logits 和 target 转换为适当的格式并计算损失。

    Args:
        num_classes (int): 类别数量，用于 one-hot 编码。
        reduction (str): 指定如何聚合损失。选项包括 'none', 'batchmean', 'sum', 'mean'。默认为 'batchmean'。
    """
    def __init__(self, num_classes, reduction='batchmean'):
        super(CustomKLDivLoss, self).__init__()
        self.num_classes = num_classes
        self.kl_div = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits, targets):
        """
        前向传播计算KL散度损失。

        Args:
            logits (Tensor): 模型的预测输出，形状为 (batch_size, num_classes)，未经过softmax归一化。
            targets (Tensor): 真实标签，形状为 (batch_size)，为类别索引。

        Returns:
            Tensor: 计算得到的KL散度损失。
        """
        # 将 logits 转换为对数概率
        log_probs = F.log_softmax(logits, dim=1)

        # 将目标标签转换为 one-hot 向量
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float()

        # 计算KL散度损失
        loss = self.kl_div(log_probs, one_hot_targets)

        return loss
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
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(
            self.ctx), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # 设置surrogate model
        self.vanilla_model = VanillaCLIP(
            cfg.MODEL.BACKBONE.NAME,
            classnames,
            cfg.TRAINER.MAPLE.PREC,
            cfg.DATASET.NAME
        ).cuda()
        self.surrogate = (
            self.vanilla_model if getattr(cfg.TRAINER.MAPLE, "SURROGATE", "vanilla_model") == "vanilla_model"
            else self)

        # 初始化各种对抗攻击方法
        self._init_attacks(cfg)

    def _init_attacks(self, cfg):
        """初始化所有可能用到的攻击方法"""
        self.attacks = {
            'pgd': PGDAttack(
                depth=cfg.TRAINER.MAPLE.PROMPT_DEPTH,
                eps=getattr(cfg.TRAINER.MAPLE, "EPSILON", 2. / 255),
                steps=getattr(cfg.TRAINER.MAPLE, "ADV_STEPS", 10)
            ),
            'fgsm': FGSMAttack(
                depth=cfg.TRAINER.MAPLE.PROMPT_DEPTH,
                eps=getattr(cfg.TRAINER.MAPLE, "EPSILON", 8. / 255),
                # steps=100
            ),
            # 可以添加更多攻击方法
            # "su":SU()
        }

    def _normalize_image(self, image):
        """标准化图像的辅助方法"""
        normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]
        )
        return normalize(image)

    def forward(self, image, label=None):
        """标准前向传播"""
        image = self._normalize_image(image)

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits

    def generate_adv(self, images, labels, attack_type='pgd'):
        """生成对抗样本"""
        if attack_type not in self.attacks:
            raise ValueError(f"Unknown attack type: {attack_type}")

        attack = self.attacks[attack_type]
        return attack.generate(
            model=self.surrogate,
            images=images,
            labels=labels,
            normalize=self._normalize_image
        )

    def forward_adv(self, image, label=None, attack_type='pgd'):
        image_adv = self.generate_adv(image, label, attack_type)
        # 使用对抗样本进行前向传播
        return self.forward(image_adv, label)

    def evaluate_robustness(self, image, label, attack_types=None):
        """评估模型在多种攻击下的鲁棒性
        Args:
            image: 输入图像
            label: 标签
            attack_types: 要评估的攻击类型列表，如果为None则评估所有已注册的攻击
        Returns:
            dict: 包含每种攻击方法下的logits
        """
        if attack_types is None:
            attack_types = list(self.attacks.keys())

        results = {}
        # 加入清洁样本的结果
        results['clean'] = self.forward(image)

        # 各种攻击方法的结果
        for attack_type in attack_types:
            results[attack_type] = self.forward_adv(image, label, attack_type)

        return results


class VanillaCLIP(nn.Module):
    def __init__(self, backbone_name, classnames, prec="fp32", dataset_name="ImageNet"):
        super().__init__()
        self.model, _ = clip_official.clip.load(backbone_name, device='cpu')
        self.model.cuda()
        self.dtype = torch.float32 if prec == "fp32" else torch.float16
        self.dataset_name = dataset_name
        self.prepare_text_features(classnames)



    def prepare_text_features(self, classnames):
        """根据数据集选择对应模板"""
        CUSTOM_TEMPLATES = {
            "OxfordPets": "a photo of a {}, a type of pet.",
            "OxfordFlowers": "a photo of a {}, a type of flower.",
            "FGVCAircraft": "a photo of a {}, a type of aircraft.",
            "DescribableTextures": "{} texture.",
            "EuroSAT": "a centered satellite photo of {}.",
            "StanfordCars": "a photo of a {}.",
            "Food101": "a photo of {}, a type of food.",
            "SUN397": "a photo of a {}.",
            "Caltech101": "a photo of a {}.",
            "UCF101": "a photo of a person doing {}.",
            "ImageNet": "a photo of a {}.",
            "ImageNetSketch": "a photo of a {}.",
            "ImageNetV2": "a photo of a {}.",
            "ImageNetA": "a photo of a {}.",
            "ImageNetR": "a photo of a {}.",
        }
        template = CUSTOM_TEMPLATES.get(
            self.dataset_name,  # 优先使用数据集对应的模板
            "a photo of a {}"  # 默认后备模板
        )
        print(f"Using template: {template}")

        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_prompts)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, image, label=None, return_logits=None):
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        image = normalize(image)
        image_features = self.model.encode_image(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.t()

        # return F.cross_entropy(logits, label)
        if return_logits:
            return logits
        return F.cross_entropy(logits, label)

class Hard_VanillaCLIP(nn.Module):
    def __init__(self, backbone_name, classnames, prec="fp32", dataset_name="ImageNet"):
        super().__init__()
        # self.model, _ = clip_official.clip.load(backbone_name, device='cpu')
        self.load_model(backbone_name)
        self.model.cuda()
        self.dtype = torch.float32 if prec == "fp32" else torch.float16
        self.dataset_name = dataset_name
        self.prepare_text_features(classnames)

    def load_model(self,backbone_name):

        # backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        self.model, _ = clip_official.clip.load(model_path,device='cpu')

    def prepare_text_features(self, classnames):
        """根据数据集选择对应模板"""
        CUSTOM_TEMPLATES = {
            "OxfordPets": "a photo of a {}, a type of pet.",
            "OxfordFlowers": "a photo of a {}, a type of flower.",
            "FGVCAircraft": "a photo of a {}, a type of aircraft.",
            "DescribableTextures": "{} texture.",
            "EuroSAT": "a centered satellite photo of {}.",
            "StanfordCars": "a photo of a {}.",
            "Food101": "a photo of {}, a type of food.",
            "SUN397": "a photo of a {}.",
            "Caltech101": "a photo of a {}.",
            "UCF101": "a photo of a person doing {}.",
            "ImageNet": "a photo of a {}.",
            "ImageNetSketch": "a photo of a {}.",
            "ImageNetV2": "a photo of a {}.",
            "ImageNetA": "a photo of a {}.",
            "ImageNetR": "a photo of a {}.",
        }
        template = CUSTOM_TEMPLATES.get(
            self.dataset_name,  # 优先使用数据集对应的模板
            "a photo of a {}"  # 默认后备模板
        )
        print(f"Using template: {template}")

        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_prompts)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, image, label=None, return_logits=True):
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        image = normalize(image)
        image_features = self.model.encode_image(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.t()

        # return F.cross_entropy(logits, label)
        if return_logits:
            return logits
        return F.cross_entropy(logits, label)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    # def __init__(self):
    #     super().__init__()
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]
        self.adv_train = getattr(cfg.TRAINER.MAPLE, "ADV_TRAIN", True)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MAPLE.PREC

        if prec == "amp":
            with autocast():
                loss = self._compute_loss(model, image, label)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = self._compute_loss(model, image, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {"loss": loss.item()}

    def _compute_loss(self, model, image, label):
        """计算训练损失（包括对抗训练）"""
        # 标准前向传播
        loss = model(image, label)

        # 对抗训练 (训练时只使用PGD)
        if self.adv_train:
            adv_images = model.generate_adv(image, label, attack_type='pgd')
            adv_loss = model(adv_images, label)
            loss = (loss + adv_loss) / 2

        return loss

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # self.vanilla_model = self.model.vanilla_model

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)



        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        if cfg.TEST.HARD_PROMPT == True:
            self.hard_model = Hard_VanillaCLIP(
                cfg.MODEL.BACKBONE.NAME,
                classnames,
                cfg.TRAINER.MAPLE.PREC,
                cfg.DATASET.NAME
            ).cuda()
    # def forward_backward(self, batch):
    #     image, label = self.parse_batch_train(batch)
    #
    #     model = self.model
    #     optim = self.optim
    #     scaler = self.scaler
    #     prec = self.cfg.TRAINER.MAPLE.PREC
    #
    #     if prec == "amp":
    #         with autocast():
    #             # 标准前向传播
    #             output = model(image, label)
    #             loss = output["loss"]
    #
    #             # 对抗训练
    #             if self.adv_train:
    #                 delta = self.model.PGD(image, label, steps=self.steps)
    #                 adv_output = model(image + delta, label)
    #                 loss = (loss + adv_output["loss"]) / 2
    #
    #         optim.zero_grad()
    #         scaler.scale(loss).backward()
    #         scaler.step(optim)
    #         scaler.update()
    #
    #     else:
    #         # 标准前向传播
    #         loss = model(image, label)
    #         # loss = output["loss"]
    #
    #         # 对抗训练
    #         if self.adv_train:
    #             delta = self.model.PGD(image, label, steps=self.steps)
    #             adv_loss = model(image + delta, label)
    #             loss = (loss + adv_loss) / 2
    #
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #
    #     if (self.batch_idx + 1) == self.num_batches:
    #         self.update_lr()
    #
    #     loss_summary = {
    #         "loss": loss.item()
    #     }
    #
    #     return loss_summary

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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def inference(self, inference_output_dir="output/debug.pkl", split=None, hard_prompt=False):
        """Generate model outputs for a given split and save them to a .pkl file."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Generate outputs on the *{split}* set")

        # file_name = f"{split}_data.pkl"

        if os.path.exists(inference_output_dir):
            # Load generated outputs from a .pkl file
            with open(inference_output_dir, "rb") as f:
                data = pickle.load(f)
            outputs = data["outputs"]
            labels = data["labels"]
        else:
            # Generate outputs and save them to a .pkl file
            outputs = []
            labels = []

            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                if hard_prompt:
                    output = self.hard_model(input, return_logits=True)
                else:
                    output = self.model_inference(input)
                outputs.append(output.detach().cpu())
                labels.append(label.detach().cpu())

            # Save generated outputs to a .pkl file
            data = {"outputs": outputs, "labels": labels}
            with open(inference_output_dir, "wb") as f:
                pickle.dump(data, f)

            print(f"Generated outputs saved to {inference_output_dir}")

        return outputs, labels

    def adv_inference(self, adv_image_dir='output/debug_adv_image.pkl',adv_inference_output_dir="output/debug_adv.pkl", attack_type="pgd",split=None, hard_prompt=False):
        """Generate model outputs for a given split and save them to a .pkl file."""
        """
        args:
            adv_inference_output_dir : 保存模型输出的inference_logits结果的路径
            split : 需要inference的split
            hard_prompt : 是否使用hard prompt 
        return:
            outputs : 模型输出的logits和label      
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Generate outputs on the *{split}* set")

        # file_name = f"{split}_data.pkl"

        if os.path.exists(adv_inference_output_dir):
            # Load generated outputs from a .pkl file
            with open(adv_inference_output_dir, "rb") as f:
                data = pickle.load(f)
            outputs = data["outputs"]
            labels = data["labels"]
        else:
            # Generate outputs and save them to a .pkl file
            outputs = []
            labels = []
            if not os.path.exists(adv_image_dir):
                ADV_IMAGES=[]
            else:
                with open(adv_image_dir,'rb') as f:
                    ADV_IMAGES=pickle.load(f)
            for batch_idx, batch in enumerate(tqdm(data_loader)):

                input, label = self.parse_batch_test(batch)
                if batch_idx>(len(ADV_IMAGES)-1):
                    adv_image=self.model.generate_adv(input,label, attack_type=attack_type)
                else:
                    adv_image=ADV_IMAGES[batch_idx].cuda()
                # adv_image=None
                if not hard_prompt:
                    output = self.model(adv_image)
                else:
                    # adv_image = self.model.generate_adv(input, label)
                    with torch.no_grad():
                        output = self.hard_model(adv_image, return_logits=True)

                outputs.append(output.detach().cpu())
                labels.append(label.detach().cpu())
                if (adv_image is not None) and (batch_idx>(len(ADV_IMAGES)-1)):
                    adv_image_cpu=adv_image.detach().cpu().clone()
                    ADV_IMAGES.append(adv_image_cpu)
                    if batch_idx % 500 == 0 or batch_idx == len(data_loader)-1:
                        with open(adv_image_dir,"wb") as f:
                            pickle.dump(ADV_IMAGES,f)
                    # del adv_image
                    torch.cuda.empty_cache()

            # Save generated outputs to a .pkl file
            data = {"outputs": outputs, "labels": labels}
            with open(adv_inference_output_dir, "wb") as f:
                pickle.dump(data, f)

            print(f"Generated outputs saved to {adv_inference_output_dir}")

        return outputs, labels

    def evaluation(self, outputs, labels, split=None):
        """Evaluate model performance using generated outputs."""
        self.evaluator.reset()

        # print(f"Evaluate on the *{split}* set")

        for output, label in zip(outputs, labels):
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def test(self,inference_output_dir="output/debug.pkl", split=None):
        output,label=self.inference(inference_output_dir=inference_output_dir,split=split)
        result=self.evaluation(output,label,split=split)

    def test_adv(self,adv_image_dir="output/debug_adv_image.pkl",adv_inference_output_dir="output/debug_adv.pkl", attack_type="pgd",split=None):
        output, label = self.adv_inference(adv_image_dir=adv_image_dir,adv_inference_output_dir=adv_inference_output_dir, attack_type=attack_type,split=split)
        result = self.evaluation(output, label, split=split)

    def test_hard_prompt(self, inference_output_dir="output/debug_hard_prompt.pkl", split=None):
        output, label = self.inference(inference_output_dir=inference_output_dir, split=split, hard_prompt=True)
        result = self.evaluation(output, label, split=split)

    def test_hard_prompt_adv(self, adv_image_dir="output/debug_adv_image.pkl",adv_inference_output_dir="output/debug_adv_hard_prompt.pkl", attack_type="pgd",split=None):
        output, label = self.adv_inference(adv_image_dir=adv_image_dir,adv_inference_output_dir=adv_inference_output_dir,attack_type=attack_type,split=split, hard_prompt=True)
        result = self.evaluation(output, label, split=split)



