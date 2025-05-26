import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
import math
import matplotlib.pyplot as plt
import wandb
import datetime
import os

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)
        
        self.wandb_flag = False
            
        
    def set_wandb(self, entity_name=None, project_name=None, group_name=None, run_name=None, additional_config=None): 
        """
        default:
        - entity_name: "jin749"
        - project_name: self.cfg.TRAINER.NAME
        - group_name: datetime.datetime.now().strftime("%Y-%m-%d")
        - run_name: f"{self.cfg.DATASET.NAME}_{self.cfg.DATASET.NUM_SHOTS}shots_seed{self.cfg.SEED}"
        
        """
        wandb_config = {
            "host": os.uname()[1],
            "model": self.cfg.MODEL.BACKBONE.NAME,
            "dataset": self.cfg.DATASET.NAME,
            "shots": self.cfg.DATASET.NUM_SHOTS,
            "seed": self.cfg.SEED,
            "class_subset": self.cfg.DATASET.SUBSAMPLE_CLASSES,
            "num_classes": len(self._lab2cname),
            "output_dir": self.cfg.OUTPUT_DIR,
            
        }
        if entity_name is None:
            entity_name = "jin749"
        if project_name is None:
            project_name = self.cfg.TRAINER.NAME
        if group_name is None:
            group_name = datetime.datetime.now().strftime("%Y-%m-%d")
        if run_name is None:
            run_name = f"{self.cfg.DATASET.NAME}_{self.cfg.DATASET.NUM_SHOTS}shots_{self.cfg.DATASET.SUBSAMPLE_CLASSES}_{self.cfg.SEED}"
        if additional_config is not None:
            wandb_config.update(additional_config)
        wandb.init(
            entity=entity_name,
            project=project_name,
            group=group_name,
            name=run_name,
            config=wandb_config
        )
        self.wandb_flag = True

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._logits_pos = [] # jin
        self._ligits_neg = [] # jin
        self._correct_base = 0 # jin
        self._total_base = 0 # jin
        self._logits_base = [] # jin
        self._correct_new = 0 # jin
        self._total_new = 0 # jin
        self._logits_new = []
        self._n_cls = len(self._lab2cname) # jin
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        logit = mo.max(1)[0] # jin
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]
        
        # if self.cfg.DATASET.SUBSAMPLE_CLASSES == "all":
        #     ###### custom ######
        #     excluding_class_num = 16
        #     ####################
        #     n_cls = len(self._lab2cname) + excluding_class_num
        #     labels = list(range(n_cls))
        #     m = math.ceil(n_cls / 2)
        #     new_labels = labels[m:]
        #     gt_new_idx = [i for i, y in enumerate(gt) if y in new_labels]
        #     self._correct_new += int(matches[gt_new_idx].sum().item())
        #     self._total_new += len(gt_new_idx)
        if self.cfg.DATASET.SUBSAMPLE_CLASSES.isdecimal() and len(self.cfg.DATASET.SUBSAMPLE_CLASSES) == 6:
            n = self.cfg.DATASET.ORIGINAL_NUM_CLASSES
            ratio = self.cfg.DATASET.SUBSAMPLE_CLASSES
            ratio1 = int(ratio[:3]) / 100
            ratio2 = int(ratio[3:]) / 100
            m = math.ceil(n / 2)
            mb = math.ceil(m * ratio1)
            mn = math.ceil((n - m) * ratio2)
            print(f"mb: {mb}, mn: {mn}")
            base_labels = list(range(mb))
            new_labels = list(range(mb, mb + mn))
            assert len(base_labels) + len(new_labels) == len(self._lab2cname)
            gt_base_idx = [i for i, y in enumerate(gt) if y in base_labels]
            gt_new_idx = [i for i, y in enumerate(gt) if y in new_labels]
            self._correct_base += int(matches[gt_base_idx].sum().item())
            self._total_base += len(gt_base_idx)
            self._correct_new += int(matches[gt_new_idx].sum().item())
            self._total_new += len(gt_new_idx)
            self._logits_base.extend(logit[gt_base_idx].data.cpu().numpy().tolist())
            self._logits_new.extend(logit[gt_new_idx].data.cpu().numpy().tolist())
            
        elif self.cfg.DATASET.SUBSAMPLE_CLASSES == "all":
            m = math.ceil(self._n_cls / 2)
            base_labels = list(range(m))
            new_labels = list(range(m, self._n_cls))
            gt_base_idx = [i for i, y in enumerate(gt) if y in base_labels]
            gt_new_idx = [i for i, y in enumerate(gt) if y in new_labels]
            self._correct_base += int(matches[gt_base_idx].sum().item())
            self._total_base += len(gt_base_idx)
            self._correct_new += int(matches[gt_new_idx].sum().item())
            self._total_new += len(gt_new_idx)    
            self._logits_base.extend(logit[gt_base_idx].data.cpu().numpy().tolist())
            self._logits_new.extend(logit[gt_new_idx].data.cpu().numpy().tolist())
            

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        # self._logits.extend(logit.data.cpu().numpy().tolist()) # jin
        # logits of matches
        self._logits_pos.extend(logit[matches.bool()].data.cpu().numpy().tolist()) # jin
        self._ligits_neg.extend(logit[~matches.bool()].data.cpu().numpy().tolist()) # jin

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        subset_list = ["base", "new", "all", "base*", "new*"]
        for subset in subset_list:
            results[subset] = None
        
        acc = 100.0 * self._correct / self._total
        results[self.cfg.DATASET.SUBSAMPLE_CLASSES] = acc
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )


        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        
        # neg, pos logits 분포 시각화 및 저장
        plt.figure(figsize=(8, 5))
        
        plt.hist(self._logits_pos, bins=30, color='red', alpha=0.5, label='Positive Logits')
        pos_mean = np.mean(self._logits_pos)
        pos_std = np.std(self._logits_pos)
        plt.axvline(pos_mean, color='red', linestyle='dashed', linewidth=1)
        plt.axvline(pos_mean + pos_std, color='red', linestyle='dotted', linewidth=1)
        plt.axvline(pos_mean - pos_std, color='red', linestyle='dotted', linewidth=1)
        
        plt.hist(self._ligits_neg, bins=30, color='blue', alpha=0.5, label='Negative Logits')
        neg_mean = np.mean(self._ligits_neg)
        neg_std = np.std(self._ligits_neg)
        plt.axvline(neg_mean, color='blue', linestyle='dashed', linewidth=1)
        plt.axvline(neg_mean + neg_std, color='blue', linestyle='dotted', linewidth=1)
        plt.axvline(neg_mean - neg_std, color='blue', linestyle='dotted', linewidth=1)

    
        plt.title("Distribution of Max Logits")
        plt.legend()
        plt.xlabel("Logit Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        # 저장 경로 지정 및 저장
        save_path = osp.join(self.cfg.OUTPUT_DIR, "logit_distribution.png")
        plt.savefig(save_path)
        print(f"Logit distribution plot saved to: {save_path}")

        plt.close() # jin

        print()
        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )
        
        print("Number of classes: ", self._n_cls)
        if (self.cfg.DATASET.SUBSAMPLE_CLASSES.isdecimal() and len(self.cfg.DATASET.SUBSAMPLE_CLASSES) == 6) or self.cfg.DATASET.SUBSAMPLE_CLASSES == "all":
            acc_base = 100.0 * self._correct_base / self._total_base if self._total_base > 0 else 0
            err_base = 100.0 - acc_base
            results["base*"] = acc_base
            results["error_rate_base*"] = err_base
            print(
                "=> result base*\n"
                f"* total: {self._total_base:,}\n"
                f"* correct: {self._correct_base:,}\n"
                f"* accuracy: {acc_base:.1f}%\n"
                f"* error: {err_base:.1f}%\n"
            )
            acc_new = 100.0 * self._correct_new / self._total_new
            err_new = 100.0 - acc_new
            results["new*"] = acc_new
            results["error_rate_new*"] = err_new
            print(
                "=> result new*\n"
                f"* total: {self._total_new:,}\n"
                f"* correct: {self._correct_new:,}\n"
                f"* accuracy: {acc_new:.1f}%\n"
                f"* error: {err_new:.1f}%\n"
            )
            # logits 분포 시각화 및 저장
            plt.figure(figsize=(8, 5))
            plt.hist(self._logits_base, bins=30, color='red', alpha=0.5, label='Base Logits')
            base_mean = np.mean(self._logits_base)
            base_std = np.std(self._logits_base)
            plt.axvline(base_mean, color='red', linestyle='dashed', linewidth=1)
            plt.axvline(base_mean + base_std, color='red', linestyle='dotted', linewidth=1)
            plt.axvline(base_mean - base_std, color='red', linestyle='dotted', linewidth=1)
            plt.hist(self._logits_new, bins=30, color='blue', alpha=0.5, label='New Logits')
            new_mean = np.mean(self._logits_new)
            new_std = np.std(self._logits_new)
            plt.axvline(new_mean, color='blue', linestyle='dashed', linewidth=1)
            plt.axvline(new_mean + new_std, color='blue', linestyle='dotted', linewidth=1)
            plt.axvline(new_mean - new_std, color='blue', linestyle='dotted', linewidth=1)
            plt.title("Distribution of Base* and New* Logits")
            plt.legend()
            plt.xlabel("Logit Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            # 저장 경로 지정 및 저장
            save_path = osp.join(self.cfg.OUTPUT_DIR, "logit_distribution_base_new.png")
            plt.savefig(save_path)
            print(f"Logit distribution plot saved to: {save_path}")
            plt.close()
            
        if self.wandb_flag:
            wandb.log({
                "self.cfg.DATASET.SUBSAMPLE_CLASSES": results["accuracy"],
                "base*": results["base*"],
                "new*": results["new*"],
            })
                    

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")
            
        self.results = results

        return results
