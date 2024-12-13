import torch
from torch.nn import functional as F
import torch.nn as nn


class BaseAttack:
    """对抗攻击的基类"""

    def __init__(self, depth=1, eps=8. / 255, steps=100, alpha=None):
        self.init_loss()
        self.eps = eps
        self.steps = steps
        self.alpha = alpha if alpha is not None else eps / steps * 1.5
        self.depth = depth

    def generate(self, model, images, labels, normalize=None):
        """生成对抗样本的接口方法"""
        raise NotImplementedError

    def init_loss(self):
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, logits, label):
        """计算损失函数的接口方法"""
        if self.depth>1:
            return F.cross_entropy(logits, label)
        elif self.depth==1:
            log_probs = F.log_softmax(logits, dim=1)
            one_hot_labels = F.one_hot(label, num_classes=logits.shape[1]).float()
            return self.kl_div(log_probs, one_hot_labels)
        # raise NotImplementedError
        else:
            raise ValueError("Invalid depth value")


class PGDAttack(BaseAttack):
    """PGD攻击实现"""

    def generate(self, model, images, labels, normalize=None):
        if normalize is not None:
            images = normalize(images)

        delta = torch.zeros_like(images).cuda()
        delta.requires_grad = True

        for _ in range(self.steps):
            if delta.grad is not None:
                delta.grad.zero_()

            logits=model(images+delta,return_logits=True)
            # loss = model(images + delta, labels)
            loss=self.compute_loss(logits,labels)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = delta + self.alpha * torch.sign(grad)
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)

        delta = delta.detach()
        return torch.clamp(images + delta, 0, 1)


# 可以添加更多攻击方法
class FGSMAttack(BaseAttack):
    """FGSM攻击实现"""

    def generate(self, model, images, labels, normalize=None):
        if normalize is not None:
            images = normalize(images)

        images.requires_grad = True
        # loss = model(images, labels)
        logits = model(images,return_logits=True)
        loss = self.compute_loss(logits, labels)
        loss.backward()

        delta = self.eps * torch.sign(images.grad.detach())
        return torch.clamp(images + delta, 0, 1)
