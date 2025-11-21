import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import re
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# ----------------------
# 回归任务配置（添加图像相关参数）
# ----------------------
regression_config = {
    "data_csv": "C:\\Users\\34841\\Desktop\\H1.csv",
    "smiles_column": "smiles",
    "image_column": "image_path",
    "target_column": "log(Jet Fire)",
    "numeric_columns": [
        "T", "P", "Leak Size", "Material Quantity",
        "Number of carbon atoms", "Number of hydrogen atoms",
        "Number of oxygen atoms", "Number of nitrogen atoms",
        "Number of sulfur atoms","Number of halogen atoms","Molecular Weight",
        "NFPA fire rating", "DM", "εHOMO", "εLUMO", "μ", "η", "ω"
    ],
    "max_seq_len": 64,
    "d_model": 768,
    "image_feature_dim": 256,
    "epochs": 500,
    "batch_size": 32,
    "freeze_transformer": False,
    "freeze_image_encoder": False,
    "lr": 1e-4,
    "transformer_lr": 1e-5,
    "image_lr": 1e-5,
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "regression_output",
    "model_path": "regression_output/best_model2.pth",
    "vocab_path": "output/222vocab.json",
    "image_encoder_path": "output/image_encoder.pth",  # 使用PyTorch预训练权重
    "test_size": 0.15,
    "val_size": 0.15,
    "patience": 20
}


# ----------------------
# SMILES分词器
# ----------------------
def smiles_tokenizer(smiles: str) -> list:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    return re.findall(pattern, smiles)


# ----------------------
# 图像预处理
# ----------------------
image_transform = transforms.Compose([
    transforms.Resize((regression_config["image_size"], regression_config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ----------------------
# 数据集类（添加图像支持）
# ----------------------
class RegressionDataset(Dataset):
    def __init__(self, encoded_data, numeric_features, targets, original_indices,
                 split_markers, image_paths, transform=None):
        assert len(encoded_data) == len(numeric_features) == len(targets) == len(image_paths), "数据维度不一致"
        self.data = encoded_data
        self.numeric = numeric_features
        self.targets = targets
        self.original_indices = original_indices
        self.split_markers = split_markers
        self.image_paths = image_paths
        self.transform = transform or image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"无法加载图像 {img_path}: {str(e)}")
            # 创建空白图像作为替代
            image = torch.zeros(3, regression_config["image_size"], regression_config["image_size"])

        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.numeric[idx].tolist(), dtype=torch.float),
            image,  # 新增：图像张量
            torch.tensor(self.targets[idx], dtype=torch.float),
            self.original_indices[idx],
            self.split_markers[idx]
        )


# ----------------------
# 数据预处理（添加图像支持）
# ----------------------
def preprocess_data(config):
    # 加载词汇表
    with open(config["vocab_path"]) as f:
        vocab = json.load(f)

    # 读取数据文件
    try:
        df = pd.read_csv(config["data_csv"], encoding='gbk').reset_index(drop=True)
    except UnicodeDecodeError:
        raise ValueError("文件编码错误！尝试使用 encoding='gbk'")

    # 检查列是否存在
    missing_columns = [col for col in config["numeric_columns"] + [config["image_column"]]]
    missing_columns = [col for col in missing_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少列: {missing_columns}")

    # 处理数值特征
    numeric_features = df[config["numeric_columns"]].values
    imputer = SimpleImputer(strategy='mean')
    numeric_features_imputed = imputer.fit_transform(numeric_features)
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features_imputed)

    # 数据有效性过滤
    valid_data, valid_numeric_scaled, valid_numeric_original = [], [], []
    valid_targets, valid_smiles, original_indices = [], [], []
    valid_image_paths = []  # 新增：存储图像路径

    for idx in range(len(df)):
        s = df.iloc[idx][config["smiles_column"]]
        mol = Chem.MolFromSmiles(s)
        numeric_row_scaled = numeric_features_scaled[idx]
        numeric_row_original = numeric_features_imputed[idx]
        image_path = df.iloc[idx][config["image_column"]]  # 新增：获取图像路径

        if mol is not None and not np.isnan(numeric_row_scaled).any() and pd.notna(image_path):
            tokens = ["[SOS]"] + smiles_tokenizer(s) + ["[EOS]"]
            ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
            padded = ids[:config["max_seq_len"]] + [0] * (config["max_seq_len"] - len(ids))
            valid_data.append(padded)
            valid_numeric_scaled.append(numeric_row_scaled)
            valid_numeric_original.append(numeric_row_original)
            valid_targets.append(df.iloc[idx][config["target_column"]])
            valid_smiles.append(s)
            original_indices.append(idx)
            valid_image_paths.append(image_path)  # 新增

    # 数据划分
    data = np.array(valid_data)
    numeric_scaled = np.array(valid_numeric_scaled)
    numeric_original = np.array(valid_numeric_original)
    targets = np.array(valid_targets)
    original_indices = np.array(original_indices)
    image_paths = np.array(valid_image_paths)  # 新增

    # 三级数据划分（添加图像路径）
    (train_val_data, test_data,
     train_val_numeric_scaled, test_numeric_scaled,
     train_val_numeric_original, test_numeric_original,
     train_val_targets, test_targets,
     train_val_indices, test_indices,
     train_val_image_paths, test_image_paths) = train_test_split(  # 新增
        data, numeric_scaled, numeric_original, targets, original_indices, image_paths,
        test_size=config["test_size"], random_state=42
    )

    (train_data, val_data,
     train_numeric_scaled, val_numeric_scaled,
     train_numeric_original, val_numeric_original,
     train_targets, val_targets,
     train_indices, val_indices,
     train_image_paths, val_image_paths) = train_test_split(  # 新增
        train_val_data, train_val_numeric_scaled, train_val_numeric_original,
        train_val_targets, train_val_indices, train_val_image_paths,
        test_size=config["val_size"], random_state=42
    )

    # 为每条数据添加数据集标记
    train_markers = ["train"] * len(train_data)
    val_markers = ["val"] * len(val_data)
    test_markers = ["test"] * len(test_data)

    train_indices = np.arange(len(train_data))
    val_indices = np.arange(len(val_data))
    test_indices = np.arange(len(test_data))

    return {
        "train": (train_data, train_numeric_scaled, train_targets),
        "val": (val_data, val_numeric_scaled, val_targets),
        "test": (test_data, test_numeric_scaled, test_targets),
        "smiles": valid_smiles,
        "original_indices": original_indices,
        "original_numeric": numeric_original,
        "train_original_numeric": train_numeric_original,
        "val_original_numeric": val_numeric_original,
        "test_original_numeric": test_numeric_original,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_markers": train_markers,
        "val_markers": val_markers,
        "test_markers": test_markers,
        "image_paths": valid_image_paths,  # 新增
        "train_image_paths": train_image_paths,  # 新增
        "val_image_paths": val_image_paths,  # 新增
        "test_image_paths": test_image_paths  # 新增
    }


# ----------------------
# 回归模型（添加图像编码器）
# ----------------------
class RegressionModel(torch.nn.Module):
    def __init__(self, pretrained_model, vocab_size, num_numeric):
        super().__init__()
        self.embedding = pretrained_model.embedding
        self.position_embedding = pretrained_model.position_embedding
        self.transformer = pretrained_model.transformer

        # 数值特征处理
        self.numeric_fc = torch.nn.Sequential(
            torch.nn.Linear(num_numeric, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.LeakyReLU(0.1)
        )

        # 图像编码器（使用ResNet50）
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, regression_config["image_feature_dim"]),
            torch.nn.ReLU()
        )

        # 冻结图像编码器（如果需要）
        if regression_config["freeze_image_encoder"]:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # 回归头（调整输入维度以包含图像特征）
        combined_dim = regression_config["d_model"] + 256 + regression_config["image_feature_dim"]
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1)
        )

        # 冻结Transformer（如果需要）
        if regression_config["freeze_transformer"]:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, x_smiles, x_numeric, x_image):
        # SMILES特征提取
        positions = torch.arange(x_smiles.size(1), device=x_smiles.device).expand(x_smiles.size(0), -1)
        x_embed = self.embedding(x_smiles) + self.position_embedding(positions)
        trans_out = self.transformer(x_embed)
        smiles_feat = trans_out[:, 0, :]

        # 数值特征提取
        numeric_feat = self.numeric_fc(x_numeric)

        # 图像特征提取
        image_feat = self.image_encoder(x_image)

        # 特征融合
        combined = torch.cat([smiles_feat, numeric_feat, image_feat], dim=1)

        return self.reg_head(combined).squeeze()


# ----------------------
# 训练流程（添加图像处理）
# ----------------------
def train_regression():
    # 初始化输出目录
    Path(regression_config["output_dir"]).mkdir(exist_ok=True)

    processed_data = preprocess_data(regression_config)

    # 创建数据集（添加图像路径）
    train_dataset = RegressionDataset(
        *processed_data["train"],
        processed_data["train_indices"],
        processed_data["train_markers"],
        processed_data["train_image_paths"],  # 新增
        transform=image_transform
    )
    val_dataset = RegressionDataset(
        *processed_data["val"],
        processed_data["val_indices"],
        processed_data["val_markers"],
        processed_data["val_image_paths"],  # 新增
        transform=image_transform
    )
    test_dataset = RegressionDataset(
        *processed_data["test"],
        processed_data["test_indices"],
        processed_data["test_markers"],
        processed_data["test_image_paths"],  # 新增
        transform=image_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=regression_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=regression_config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=regression_config["batch_size"], shuffle=False)

    # 加载预训练模型
    from yu import TransformerModel
    with open(regression_config["vocab_path"]) as f:
        vocab_size = len(json.load(f))
    pretrained_model = TransformerModel(vocab_size)
    pretrained_model.load_state_dict(torch.load("output/222pretrained_model.pth"))

    # 初始化模型
    model = RegressionModel(
        pretrained_model,
        vocab_size,
        num_numeric=len(regression_config["numeric_columns"])
    ).to(regression_config["device"])

    # 优化器设置（添加图像编码器参数）
    optimizer_grouped_parameters = [
        {"params": model.transformer.parameters(), "lr": regression_config["transformer_lr"]},
        {"params": model.image_encoder.parameters(), "lr": regression_config["image_lr"]},  # 新增
        {"params": model.numeric_fc.parameters(), "lr": regression_config["lr"]},
        {"params": model.reg_head.parameters(), "lr": regression_config["lr"]}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = torch.nn.MSELoss()

    # 训练监控
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # 训练循环（添加图像处理）
    for epoch in range(regression_config["epochs"]):
        model.train()
        train_loss = 0.0

        # 训练阶段
        for smiles, numeric_feat, images, labels, original_indices, split_type in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}"):
            # 将数据移至设备
            smiles = smiles.to(regression_config["device"])
            numeric_feat = numeric_feat.to(regression_config["device"])
            images = images.to(regression_config["device"])
            labels = labels.to(regression_config["device"])

            optimizer.zero_grad()
            outputs = model(smiles, numeric_feat, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for smiles, numeric_feat, images, labels, original_idx, split_type in val_loader:
                smiles = smiles.to(regression_config["device"])
                numeric_feat = numeric_feat.to(regression_config["device"])
                images = images.to(regression_config["device"])
                labels = labels.to(regression_config["device"])

                outputs = model(smiles, numeric_feat, images)
                val_loss += criterion(outputs, labels).item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['lr'].append(optimizer.param_groups[0]['lr'])

        # 学习率调整
        scheduler.step(val_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), regression_config["model_path"])
            print(f"💾 保存最佳模型（验证损失: {val_loss:.4f}）")
        else:
            patience_counter += 1
            if patience_counter >= regression_config["patience"]:
                print(f"⏹ 早停触发于第 {epoch + 1} 轮，最佳验证损失: {best_val_loss:.4f}")
                break

        # 日志输出
        print(f"\nEpoch {epoch + 1}/{regression_config['epochs']}")
        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

    # 加载最佳模型
    model.load_state_dict(torch.load(regression_config["model_path"]))
    model.eval()

    # 计算训练集、验证集和测试集的指标并保存预测结果
    for dataset_name, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
        preds, labels, original_indices = [], [], []
        numeric_features_list = []
        image_paths_list = []  # 新增

        with torch.no_grad():
            for smiles, numeric_feat, images, labels_batch, original_idx, split_type in loader:
                smiles = smiles.to(regression_config["device"])
                numeric_feat = numeric_feat.to(regression_config["device"])
                images = images.to(regression_config["device"])

                outputs = model(smiles, numeric_feat, images)
                preds.extend(outputs.detach().cpu().numpy().flatten())
                labels.extend(labels_batch.cpu().numpy().flatten())
                original_indices.extend(original_idx.cpu().numpy().flatten())

                # 根据数据集类型选择正确的原始数值
                if dataset_name == "train":
                    numeric_features_list.extend(
                        processed_data["train_original_numeric"][i]
                        for i in original_idx.cpu().numpy().flatten()
                    )
                    image_paths_list.extend(
                        processed_data["train_image_paths"][i]
                        for i in original_idx.cpu().numpy().flatten()
                    )
                elif dataset_name == "val":
                    numeric_features_list.extend(
                        processed_data["val_original_numeric"][i]
                        for i in original_idx.cpu().numpy().flatten()
                    )
                    image_paths_list.extend(
                        processed_data["val_image_paths"][i]
                        for i in original_idx.cpu().numpy().flatten()
                    )
                else:
                    numeric_features_list.extend(
                        processed_data["test_original_numeric"][i]
                        for i in original_idx.cpu().numpy().flatten()
                    )
                    image_paths_list.extend(
                        processed_data["test_image_paths"][i]
                        for i in original_idx.cpu().numpy().flatten()
                    )

        # 计算指标
        mse = mean_squared_error(labels, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, preds)
        print(f"\n✅ {dataset_name}集预测结果指标:")
        print(f"R²: {r2:.4f}|MSE: {mse:.4f} | RMSE: {rmse:.4f}")

        # 保存结果时包含原始行号和图像路径
        results_df = pd.DataFrame({
            '原始行号': original_indices,
            'SMILES': [processed_data["smiles"][i] for i in original_indices],
            '图像路径': image_paths_list,  # 新增
            **{col: [x[i] for x in numeric_features_list] for i, col in
               enumerate(regression_config["numeric_columns"])},
            '实际值': labels,
            '预测值': preds,
            '数据集来源': dataset_name
        })
        results_df.to_csv(Path(regression_config["output_dir"]) / f"{dataset_name}_results1.csv", index=False)

    # 预测整个数据集
    all_preds, all_labels, all_indices, all_split_types = [], [], [], []
    all_smiles = []
    all_numeric_features = []
    all_image_paths = []  # 新增

    for dataset_name, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
        for smiles, numeric_feat, images, labels, original_idx, split_type in loader:
            smiles = smiles.to(regression_config["device"])
            numeric_feat = numeric_feat.to(regression_config["device"])
            images = images.to(regression_config["device"])

            outputs = model(smiles, numeric_feat, images)
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_indices.extend(original_idx.cpu().numpy().flatten())
            all_split_types.extend([split_type] * len(labels))
            all_smiles.extend([processed_data["smiles"][i] for i in original_idx])

            # 获取原始数值特征
            if dataset_name == "train":
                numeric_features = processed_data["train_original_numeric"][original_idx]
                image_paths = processed_data["train_image_paths"][original_idx]  # 新增
            elif dataset_name == "val":
                numeric_features = processed_data["val_original_numeric"][original_idx]
                image_paths = processed_data["val_image_paths"][original_idx]  # 新增
            else:
                numeric_features = processed_data["test_original_numeric"][original_idx]
                image_paths = processed_data["test_image_paths"][original_idx]  # 新增

            all_numeric_features.extend(numeric_features)
            all_image_paths.extend(image_paths)  # 新增

    # 保存完整结果
    all_results_df = pd.DataFrame({
        '原始行号': all_indices,
        'SMILES': all_smiles,
        '图像路径': all_image_paths,  # 新增
        **{col: [x[i] for x in all_numeric_features] for i, col in enumerate(regression_config["numeric_columns"])},
        '实际值': all_labels,
        '预测值': all_preds,
        '数据集来源': all_split_types
    })
    all_results_df.to_csv(Path(regression_config["output_dir"]) / "all_results1.csv", index=False)

    # 计算整个数据集的指标
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    print(f"\n✅ 整个数据集的预测结果指标:")
    print(f"R²: {r2:.4f}|MSE: {mse:.4f} | RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train_regression()