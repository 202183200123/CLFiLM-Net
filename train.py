import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vehicle_split import create_vehicle_split, validate_vehicle_split, save_split_record
from sklearn.preprocessing import LabelEncoder, label_binarize
# 引入基础指标以及画图所需的 AUC、PR 曲线相关库
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score
import math
import os
import copy
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    raise ImportError("请安装 transformers: pip install transformers")

try:
    import openpyxl
except ImportError:
    print("警告: 建议安装 openpyxl 以支持 .xlsx 读取")


# =========================================================================
# Part 0: Focal Loss 定义
# =========================================================================

class FocalLoss(nn.Module):
    """
    多分类 Focal Loss 实现
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =========================================================================
# Part 1: CLFiLM-Net 模型定义
# =========================================================================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))


class BottleneckWithAttention(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None):
        super(BottleneckWithAttention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.sa(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet_SA_StrategyB(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_SA_StrategyB, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def resnet50_sa_strategy_b():
    return ResNet_SA_StrategyB(BottleneckWithAttention, [3, 4, 6, 3])


class RoadTextEncoder(nn.Module):
    def __init__(self):
        super(RoadTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
        self.fc = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, ids, mask):
        return self.fc(self.bert(ids, attention_mask=mask).pooler_output)


class TrajectoryEncoder(nn.Module):
    def __init__(self):
        super(TrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(4, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class SpatiotemporalSemanticStream(nn.Module):
    def __init__(self):
        super(SpatiotemporalSemanticStream, self).__init__()
        self.txt = RoadTextEncoder()
        self.traj = TrajectoryEncoder()
        self.gate = nn.Sequential(nn.Linear(128, 128), nn.Sigmoid())
        self.bias = nn.Sequential(nn.Linear(128, 128), nn.Tanh())

    def forward(self, traj, ids, mask):
        h_t = self.traj(traj)
        h_x = self.txt(ids, mask)
        return h_t * self.gate(h_x) + self.bias(h_x)


# 非对称双向闭环特征调制与分类
class RF_FiLM_ContextNet(nn.Module):
    def __init__(self, num_classes):
        super(RF_FiLM_ContextNet, self).__init__()
        self.vis = resnet50_sa_strategy_b()  # 视觉特征提取 [B, 2048]
        self.env = SpatiotemporalSemanticStream()  # 环境特征提取 [B, 128]

        # 创新维度一（正向）：环境特征指导视觉特征重标定
        self.forward_film_gen = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048 * 2)  # 生成用于视觉分支的 Gamma 和 Beta
        )

        # FiLM 参数生成器采用零初始化，使初始仿射变换为恒等映射
        nn.init.zeros_(self.forward_film_gen[-1].weight)
        nn.init.zeros_(self.forward_film_gen[-1].bias)

        # 创新维度一（反馈）：视觉特征校验环境特征置信度
        self.feedback_gate = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.5)
        # 输入维度: 经过反馈校验的环境(128) + 经过正向调制的视觉(2048) = 2176
        self.cls = nn.Linear(2176, num_classes)

    def forward_features(self, img, traj, ids, mask):
        # 1. 提取独立模态特征
        v_raw = self.vis(img)  # 原始视觉特征 [B, 2048]
        s_raw = self.env(traj, ids, mask)  # 原始环境特征 [B, 128]

        # 2. 正向调制 (Environment -> Vision)
        film_params = self.forward_film_gen(s_raw)
        gamma, beta = torch.split(film_params, 2048, dim=1)

        # 使用 Tanh 限制乘性调制参数的幅度
        gamma = torch.tanh(gamma)

        v_prime = (1 + gamma) * v_raw + beta  # 得到具备环境感知的视觉特征

        # 3. 反向校验 (Vision -> Environment)
        g_feedback = self.feedback_gate(v_prime)

        # 残差反馈保留环境表示并加入视觉反馈
        s_refined = s_raw + s_raw * g_feedback  # 得到经过视觉校验的环境特征

        # 4. 双向闭环融合
        final_feat = torch.cat([s_refined, v_prime], dim=1)
        return self.dropout(final_feat)

    def forward(self, img, traj, ids, mask):
        final_feat = self.forward_features(img, traj, ids, mask)
        logits = self.cls(final_feat)
        return logits, final_feat


# =========================================================================
# Part 2: Dataset
# =========================================================================

class HazMatDataset(Dataset):
    def __init__(self, file_path, img_dir, tokenizer, seq_len=5, partition="train", split_plan=None, split_seed=42):
        self.seq_len = seq_len
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"正在读取表格: {file_path}")
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python', on_bad_lines='skip')
            except:
                df = pd.read_csv(file_path, encoding='gb18030', sep=None, engine='python', on_bad_lines='skip')

        df.columns = df.columns.str.strip()
        if '具体路段' in df.columns:
            df = df.rename(columns={'具体路段': 'Location_Text'})
        elif 'Unnamed: 10' in df.columns:
            df = df.rename(columns={'Unnamed: 10': 'Location_Text'})
        elif len(df.columns) > 10:
            df = df.rename(columns={df.columns[10]: 'Location_Text'})

        df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
        df = df.sort_values(by=['车牌号', '时间']).dropna(subset=['预警类型', 'Location_Text'])
        self.le = LabelEncoder()
        df['label_idx'] = self.le.fit_transform(df['预警类型'])
        self.classes = self.le.classes_

        df['hour'] = df['时间'].dt.hour
        df['h_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['h_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['lat_n'] = (df['纬度'] - df['纬度'].mean()) / (df['纬度'].std() + 1e-5)
        df['lon_n'] = (df['经度'] - df['经度'].mean()) / (df['经度'].std() + 1e-5)

        # Assign vehicles to partitions before constructing sliding windows.
        if partition not in ('train', 'val', 'test'):
            raise ValueError('partition must be train, val or test')
        eligible = [plate for plate, group in df.groupby('车牌号') if len(group) >= seq_len]
        self.split_plan = create_vehicle_split(df, seq_len, split_seed) if split_plan is None else split_plan
        validate_vehicle_split(self.split_plan, eligible)
        self.partition = partition
        self.vehicle_ids = tuple(self.split_plan[partition])
        partition_frame = df[df['车牌号'].isin(self.vehicle_ids)]

        self.samples = []
        for plate, group in partition_frame.groupby('车牌号'):
            if len(group) < seq_len:
                continue
            vals = group[['lon_n', 'lat_n', 'h_sin', 'h_cos']].values
            texts = group['Location_Text'].values
            imgs = group['Image_Filename'].values
            lbls = group['label_idx'].values
            for i in range(len(group) - seq_len + 1):
                self.samples.append({
                    'traj': vals[i:i + seq_len], 'text': str(texts[i + seq_len - 1]),
                    'img_name': imgs[i + seq_len - 1], 'label': lbls[i + seq_len - 1], 'plate': plate
                })
        print(f"{partition}: 车辆数 {len(self.vehicle_ids)}, 窗口样本数 {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            img = Image.open(os.path.join(self.img_dir, item['img_name'])).convert('RGB')
            img = self.transform(img)
        except:
            img = torch.zeros(3, 224, 224)

        traj = torch.tensor(item['traj'], dtype=torch.float32)
        encoded = self.tokenizer(item['text'], padding='max_length', max_length=32, truncation=True,
                                 return_tensors='pt')
        return img, traj, encoded['input_ids'][0], encoded['attention_mask'][0], torch.tensor(item['label'],
                                                                                              dtype=torch.long)


# =========================================================================
# Part 4: 主控逻辑 (Train 60% -> Val 20% -> Test 20%)
# =========================================================================

if __name__ == '__main__':
    # ！！！请确保这里的路径是你本地实际存在的路径！！！
    CSV_FILE = r'C:\new_database\result\output\new_data666.xlsx'
    IMG_DIR = r'C:\new_database\result\output\new'
    BATCH_SIZE = 10
    EPOCHS_DL = 10
    LR = 1e-4
    WEIGHT_SAVE_PATH = 'base_tanh_best_weight.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 运行端到端训练策略 (创新点: RF-FiLM 双向闭环反馈 + 约束初始化) ===")
    print(f"=== 将自动保存具有最高 Val F1 的最优权重到 {WEIGHT_SAVE_PATH} ===")
    print(f"=== 自动绘制 AUC 与 PR 曲线 ===")
    print(f"=== 运行设备: {device} ===")

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print(">>> 先按车辆划分，再在各集合内构造窗口 (车辆数目标比例 60% / 20% / 20%)...")
    train_dataset = HazMatDataset(CSV_FILE, IMG_DIR, tokenizer, seq_len=5, partition='train', split_seed=42)
    val_dataset = HazMatDataset(CSV_FILE, IMG_DIR, tokenizer, seq_len=5, partition='val',
                                split_plan=train_dataset.split_plan)
    test_dataset = HazMatDataset(CSV_FILE, IMG_DIR, tokenizer, seq_len=5, partition='test',
                                 split_plan=train_dataset.split_plan)
    full_dataset = train_dataset  # Shared class mapping for evaluation
    report = save_split_record('vehicle_split_record.json', CSV_FILE,
                              {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}, seed=42)
    for partition, details in report['partitions'].items():
        if details['missing_classes']:
            print(f"提示: {partition} 中未出现这些类别: {details['missing_classes']}；请结合类别覆盖解释指标。")
    print(f"划分完成 -> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    print("划分记录已保存至 vehicle_split_record.json；窗口数量比例可能不同于车辆数量比例。")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 实例化多模态闭环融合模型
    model = RF_FiLM_ContextNet(num_classes=len(full_dataset.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = FocalLoss(gamma=2.0, alpha=None)

    # ---------------------------------------------------------------------
    # Stage 1: 深度学习主干训练 (Adam)
    # ---------------------------------------------------------------------
    print(f"\n>>> 深度学习模型训练 (Training RF-FiLM-ContextNet with Adam)...")

    history = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_val_f1 = 0.0

    for epoch in range(EPOCHS_DL):
        model.train()
        train_loss = train_correct = train_total = 0
        for step, (imgs, trajs, ids, masks, labels) in enumerate(train_loader):
            imgs, trajs, ids, masks, labels = imgs.to(device), trajs.to(device), ids.to(device), masks.to(
                device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(imgs, trajs, ids, masks)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total

        model.eval()
        val_loss = val_correct = val_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, trajs, ids, masks, labels in val_loader:
                imgs, trajs, ids, masks, labels = imgs.to(device), trajs.to(device), ids.to(device), masks.to(
                    device), labels.to(device)

                logits, _ = model(imgs, trajs, ids, masks)
                loss = loss_fn(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(avg_val_loss)

        test_loss = test_correct = test_total = 0
        with torch.no_grad():
            for imgs, trajs, ids, masks, labels in test_loader:
                imgs, trajs, ids, masks, labels = imgs.to(device), trajs.to(device), ids.to(device), masks.to(
                    device), labels.to(device)

                logits, _ = model(imgs, trajs, ids, masks)
                loss = loss_fn(logits, labels)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_correct / test_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['test_acc'].append(avg_test_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS_DL} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} | Val F1: {val_f1:.4f}")

        # ================== 按验证集 Macro-F1 保存模型 ==================
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), WEIGHT_SAVE_PATH)
            print(f"[*] 发现更优模型 (Val F1: {val_f1:.4f})，已保存权重至 '{WEIGHT_SAVE_PATH}'")
        # ==============================================================

    # ---------------------------------------------------------------------
    # Stage 2: 在独立测试集 (Test Set) 上进行最终盲测评估并绘制曲线
    # ---------------------------------------------------------------------
    print("\n>>> 训练结束，正在加载最优权重并在独立测试集 (Test Set) 上进行学术盲测及绘制曲线...")

    if os.path.exists(WEIGHT_SAVE_PATH):
        model.load_state_dict(torch.load(WEIGHT_SAVE_PATH, map_location=device))
        print(f"✅ 成功加载最优验证集表现的模型权重进行最终测试。")


    def evaluate_and_plot_curves(loader, eval_model, num_classes, class_names):
        eval_model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for imgs, trajs, ids, masks, labels in loader:
                imgs, trajs, ids, masks = imgs.to(device), trajs.to(device), ids.to(device), masks.to(device)
                logits, _ = eval_model(imgs, trajs, ids, masks)

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        y_test_bin = label_binarize(all_labels, classes=range(num_classes))
        all_probs = np.array(all_probs)
        colors = plt.cm.get_cmap('tab10', num_classes)

        # ===== 1. 绘制并保存 AUC(ROC) 曲线 =====
        plt.figure(figsize=(8, 6), dpi=300)
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors(i), lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('RF-FiLM 模型 -tanh ROC / AUC 曲线', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig('RF-FiLM模型-tanh_AUC曲线.png', dpi=300)
        plt.close()

        # ===== 2. 绘制并保存 PR 曲线 =====
        plt.figure(figsize=(8, 6), dpi=300)
        for i in range(num_classes):
            precision, recall_curve_vals, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
            ap = average_precision_score(y_test_bin[:, i], all_probs[:, i])
            plt.plot(recall_curve_vals, precision, color=colors(i), lw=2, label=f'{class_names[i]} (AP = {ap:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('RF-FiLM 模型 - Precision-Recall (PR) 曲线', fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig('RF-FiLM模型-PR_tanh曲线.png', dpi=300)
        plt.close()

        print("\n✅ ROC/AUC 曲线已保存至: RF-FiLM模型-AUC曲线.png")
        print("✅ PR 曲线已保存至: RF-FiLM模型-tanhPR曲线.png")

        return acc, f1, recall


    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    test_acc_final, test_f1_final, test_recall_final = evaluate_and_plot_curves(test_loader, model, num_classes,
                                                                                class_names)

    print("\n" + "=" * 50)
    print("[最终测试集评估指标 (Test Set Performance)]")
    print("-" * 50)
    print(f"{'Accuracy (准确率)':<20} | {test_acc_final:.4f}")
    print(f"{'Macro-F1 (F1分数)':<20} | {test_f1_final:.4f}")
    print(f"{'Macro-Recall (召回率)':<18} | {test_recall_final:.4f}")
    print("=" * 50)

    # ---------------------------------------------------------------------
    # 绘图部分：收敛曲线图
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # 左图：Loss 收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS_DL + 1), history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(range(1, EPOCHS_DL + 1), history['test_loss'], 'r--s', label='Test Loss')
    plt.title('RF-FiLM_tanh模型收敛曲线 (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend()

    # 右图：Accuracy 收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS_DL + 1), history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(range(1, EPOCHS_DL + 1), history['test_acc'], 'r--s', label='Test Acc')
    plt.title('RF-FiLM_tanh 模型准确率曲线 (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend()

    plt.tight_layout()
    plt.savefig('RF-FiLM_ContextNet_收敛曲线.png', dpi=300)
    print("✅ 收敛曲线图已成功生成并保存至: RF-FiLM_ContextNet_tanh收敛曲线.png")
