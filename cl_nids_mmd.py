import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import *

import argparse
import warnings
import math
import os
import csv

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='unsw')
parser.add_argument("--epochs", type=int, default=4)  # 离线初始训练的总轮次
parser.add_argument("--epoch_1", type=int, default=1)  # 在线持续学习阶段每个批次的训练轮次
parser.add_argument("--percent", type=float, default=0.8)  # 流式数据占比，1-percent为初始训练集占比
parser.add_argument("--sample_interval", type=int, default=20000)  # 在线学习时每个批次的样本数量
parser.add_argument("--cuda", type=str, default="0")
parser.add_argument("--num_labeled_sample", type=int, default=200)  # 每次筛选的带标签代表性样本数量
parser.add_argument("--new_sample_weight", type=float, default=100.0)
parser.add_argument("--mmd_sigma", type=float, default=1.0)  # MMD核函数的带宽参数
parser.add_argument("--drift_threshold", type=float, default=0.001)  # 漂移检测阈值（MMD统计量）
parser.add_argument("--max_candidates", type=int, default=8000)  # 梯度匹配核心集选择的最大候选样本数
parser.add_argument("--epoch_1_no_drift", type=int, default=1)  # 无漂移情况下的训练轮次
parser.add_argument("--ablate", type=str, default="none", 
                    choices=["none", "no_drift", "no_gradient_matching", "no_strategic_forgetting", "no_regularization",
                             "only_mmd", "only_gradmatch"],
                    help="Ablation: none (full), no_drift, no_gradient_matching, no_strategic_forgetting, no_regularization, "
                         "only_mmd (MMD drift + mask-based selection), only_gradmatch (Grad-Match + no drift)")
parser.add_argument("--save_curve_csv", action="store_true",
                    help="Save continual-learning curve CSV (5-seed aggregated)")
parser.add_argument("--log_curve_per_batch", action="store_true",
                    help="Evaluate full test-set F1 after each stream batch (slower, for real curve)")
parser.add_argument("--curve_csv_path", type=str, default="",
                    help="Optional output path for aggregated curve CSV")

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
sample_interval = args.sample_interval
cuda_num = args.cuda
num_labeled_sample = args.num_labeled_sample
new_sample_weight = args.new_sample_weight
mmd_sigma = args.mmd_sigma
drift_threshold = args.drift_threshold
max_candidates = args.max_candidates
epoch_1_no_drift = args.epoch_1_no_drift
ablate = args.ablate
save_curve_csv = args.save_curve_csv
log_curve_per_batch = args.log_curve_per_batch
curve_csv_path = args.curve_csv_path

seed = 5011
seed_round = 5
knowledge_distillation_weight = 0.5  # 知识蒸馏损失权重（LwF方法）

temperature = 0.02  # 对比学习的温度参数
batch_size = 128

# 根据数据集设置输入维度
if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196


if dataset == 'nsl':
    KDDTrain_dataset_path   = "NSL_pre_data/PKDDTrain+.csv"
    KDDTest_dataset_path    = "NSL_pre_data/PKDDTest+.csv"

    KDDTrain   =  load_data(KDDTrain_dataset_path)
    KDDTest    =  load_data(KDDTest_dataset_path)

    splitter_nsl = SplitData(dataset='nsl')
else:
    UNSWTrain_dataset_path   = "UNSW_pre_data/UNSWTrain.csv"
    UNSWTest_dataset_path    = "UNSW_pre_data/UNSWTest.csv"

    UNSWTrain   =  load_data(UNSWTrain_dataset_path)
    UNSWTest    =  load_data(UNSWTest_dataset_path)

    splitter_unsw = SplitData(dataset='unsw')

device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")
contrastive_loss_fn = InfoNCELoss(device, temperature)

if dataset != 'nsl':
    classification_criterion = nn.BCELoss(reduction='none')
    class_weights = None  # 类别权重将在训练数据加载后计算

# 用于汇总每个 seed 的曲线
all_seed_curves = []  # list of list[(step, f1_percent)]

for i in range(seed_round):
    current_seed = seed + i
    setup_seed(current_seed)
    print(f"Current seed: {current_seed}")

    # 数据预处理和转换
    if dataset == 'nsl':
        x_train, y_train = splitter_nsl.transform(KDDTrain, labels='labels2')
        x_test, y_test = splitter_nsl.transform(KDDTest, labels='labels2')
    else:
        x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
        x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

    x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
    x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)
    print(f'训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}')
    
    # 计算类别权重以处理类别不平衡问题
    if dataset != 'nsl':
        unique_labels, counts = torch.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        class_weights = torch.zeros(2, device=device)
        for label, count in zip(unique_labels, counts):
            # 使用逆频率作为权重：总样本数 / (类别数 * 该类别样本数)
            class_weights[label] = total_samples / (len(unique_labels) * count.item())
        abnormal_count = counts[1].item() if len(counts) > 1 else 0
        abnormal_weight = class_weights[1].item() if len(class_weights) > 1 else 0.0
        print(f'训练集类别分布: 正常={counts[0].item()}, 异常={abnormal_count}')
        print(f'类别权重: 正常={class_weights[0].item():.4f}, 异常={abnormal_weight:.4f}')
    
    torch.cuda.empty_cache()
    
    # 将训练集划分为初始训练集和流式数据池
    # initial_train: 用于离线初始训练（占比1-percent）
    # stream_data: 流式数据的第一部分（占比percent），后续将与测试集合并
    initial_train_x, stream_data_x, initial_train_y, stream_data_y = train_test_split(
        x_train, y_train, test_size=percent
    )
    
    # 计算持续学习的内存缓冲区大小
    memory_buffer_size = math.floor(x_train.shape[0] * (1 - percent))
    print(f'内存缓冲区大小: {memory_buffer_size}')
    
    # 创建初始离线训练的数据加载器
    initial_train_dataset = TensorDataset(initial_train_x, initial_train_y)
    initial_train_loader = DataLoader(
        dataset=initial_train_dataset, batch_size=batch_size, shuffle=True
    )

    # 初始化模型和教师模型（用于知识蒸馏）
    if dataset == 'nsl':
        model = AE(input_dim).to(device)
        teacher_model = AE(input_dim).to(device)
    else:
        model = AE_classifier(input_dim).to(device)
        teacher_model = AE_classifier(input_dim).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # 离线初始训练阶段：学习基础的异常检测能力
    model.train()
    for epoch in range(epochs):
        if epoch % 50 == 0 or epoch == 0:
            print(f'seed = {seed+i}, 初始训练轮次: {epoch}')
        for batch_data in initial_train_loader:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if dataset == 'nsl':
                features, recon_vec = model(inputs)
            else:
                features, recon_vec, classifications = model(inputs)
            
            # 计算对比损失（基于重构误差）
            contrastive_loss = contrastive_loss_fn(recon_vec, labels)
            
            if dataset == 'nsl':
                loss = contrastive_loss.mean()
            else:
                classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                # 应用类别权重处理不平衡数据
                if class_weights is not None:
                    sample_weights = class_weights[labels.long()]
                    classification_loss = classification_loss * sample_weights
                loss = contrastive_loss.mean() + classification_loss.mean()

            loss.backward()
            optimizer.step()
    
    # 将学生模型参数复制到教师模型（用于知识蒸馏）
    teacher_model.load_state_dict(model.state_dict())
    
    # 将数据转移到指定设备
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    initial_train_x = initial_train_x.to(device)
    initial_train_y = initial_train_y.to(device)
    
    # 初始化训练缓冲区和流式数据池
    # training_buffer: 当前训练缓冲区（将用选中的样本更新）
    # stream_data_pool: 流式数据池（将分批处理）
    training_buffer_x = initial_train_x.clone()
    training_buffer_y = initial_train_y.clone()
    stream_data_pool_x = stream_data_x.clone().to(device)
    stream_data_pool_y = stream_data_y.clone()
    
    # 打乱测试集以增加随机性
    test_permutation = torch.randperm(x_test.size(0))
    x_test = x_test[test_permutation]
    y_test = y_test[test_permutation]
    
    # 将测试集合并到流式数据池
    stream_data_pool_x = torch.cat((stream_data_pool_x, x_test), dim=0)
    stream_data_pool_y = torch.cat((stream_data_pool_y, y_test), dim=0)
    print(f'流式数据池大小: {stream_data_pool_x.shape}')
    
    # 评估初始模型性能（基线性能）
    if dataset == 'nsl':
        # 计算正常样本的重构模板（NSL数据集）
        normal_samples_mask = (initial_train_y == 0).squeeze()
        normal_reconstruction_template = torch.mean(
            F.normalize(model(initial_train_x[normal_samples_mask])[1], p=2, dim=1), dim=0
        )
        y_pred_before_continual = evaluate(
            normal_reconstruction_template, initial_train_x, initial_train_y, 
            stream_data_pool_x, 0, model
        )
        y_pred_before_continual = y_pred_before_continual.cpu().numpy()
        print('--------------------------- 持续学习前性能评估 -----------------------------------')
        test_labels = stream_data_pool_y[-x_test.shape[0]:].numpy()
        test_predictions = y_pred_before_continual[-x_test.shape[0]:].astype("int32")
        performance_before = score_detail(test_labels, test_predictions)
    else:
        print('--------------------------- 持续学习前性能评估 -----------------------------------')
        initial_test_dataset = TensorDataset(x_test, y_test)
        initial_test_loader = DataLoader(dataset=initial_test_dataset, batch_size=batch_size, shuffle=False)
        performance_before = evaluate_classifier(model, initial_test_loader, device)

    # 记录该 seed 的曲线：step=0 对应持续学习前
    seed_curve = []
    before_f1 = float(performance_before[3]) * 100.0
    seed_curve.append((0, before_f1))

####################### 开始在线持续学习 #######################
    model = model.to(device)

    batch_count = 0
    start_idx = 0
    all_batch_predictions = torch.empty(0, dtype=torch.long, device=device)  # 存储所有批次的预测结果
    all_selected_indices = []  # 存储所有批次中被选中样本的全局索引

    while start_idx < len(stream_data_pool_x):
        print(f'seed = {seed+i}, 批次 = {batch_count}')
        batch_count += 1
        
        # 从流式数据池中提取当前批次
        end_idx = min(start_idx + sample_interval, len(stream_data_pool_x))
        current_batch_x = stream_data_pool_x[start_idx:end_idx]
        current_batch_y = stream_data_pool_y[start_idx:end_idx]

        model = model.to(device)
        current_batch_y = current_batch_y.to(device)
        start_idx += sample_interval
        
        # 检测当前批次是否发生概念漂移
        if dataset == 'nsl':
            # 计算正常样本的重构模板（模型更新后需重新计算）
            buffer_normal_mask = (training_buffer_y == 0).squeeze()
            normal_reconstruction_template = torch.mean(
                F.normalize(model(training_buffer_x[buffer_normal_mask])[1], p=2, dim=1), dim=0
            )
            
            # 评估缓冲区和新批次，获取概率分布
            buffer_normal_pdf, buffer_abnormal_pdf, buffer_similarities = evaluate(
                normal_reconstruction_template, training_buffer_x, training_buffer_y,
                training_buffer_x, training_buffer_y, model, get_probs=True
            )
            new_batch_normal_pdf, new_batch_abnormal_pdf, new_batch_similarities = evaluate(
                normal_reconstruction_template, training_buffer_x, training_buffer_y,
                current_batch_x, 0, model, get_probs=True
            )
            
            # 计算正常概率占比
            buffer_normal_prob_ratio = buffer_normal_pdf / (buffer_normal_pdf + buffer_abnormal_pdf)
            new_batch_normal_prob_ratio = new_batch_normal_pdf / (new_batch_normal_pdf + new_batch_abnormal_pdf)

            drift_detected, drift_metric_value = detect_drift(
                new_batch_normal_prob_ratio, buffer_normal_prob_ratio, 
                sample_interval, drift_threshold, method='mmd', sigma=mmd_sigma
            )
            if batch_count == 1:
                print(f"使用MMD漂移阈值: {drift_threshold} (MMD值通常在0.0001-0.01范围内)")

            # 计算异常得分（值越大表示越异常/越重要）
            buffer_anomaly_scores = (1 - buffer_normal_prob_ratio).detach().view(-1)
            new_batch_anomaly_scores = (1 - new_batch_normal_prob_ratio).detach().view(-1)
        else:
            # 获取分类logits用于漂移检测（UNSW数据集）
            with torch.no_grad():
                _, _, new_batch_logits = model(current_batch_x)
                new_batch_logits = new_batch_logits.squeeze()
                _, _, buffer_logits = model(training_buffer_x)
                buffer_logits = buffer_logits.squeeze()
            
            drift_detected, drift_metric_value = detect_drift(
                new_batch_logits, buffer_logits, sample_interval, 
                drift_threshold, method='mmd', sigma=mmd_sigma
            )
            if batch_count == 1:
                print(f"使用MMD漂移阈值: {drift_threshold} (MMD值通常在0.0001-0.01范围内)")

            # 使用分类logits作为重要性得分
            buffer_anomaly_scores = buffer_logits.detach().view(-1)
            new_batch_anomaly_scores = new_batch_logits.detach().view(-1)

        # 消融实验处理
        # 消融：w/o drift detection — 不做漂移检测，每批强制视为无漂移
        if ablate == "no_drift":
            drift_detected = False
            drift_metric_value = 0.0
            if batch_count == 1:
                print("[Ablation] w/o drift detection: forcing drift_detected=False every batch.")
        
        # 消融：only_gradmatch — 仅使用梯度匹配，不做漂移检测（强制无漂移）
        if ablate == "only_gradmatch":
            drift_detected = False
            drift_metric_value = 0.0
            if batch_count == 1:
                print("[Ablation] only_gradmatch: using gradient-matching coreset selection without drift detection.")

        # 判断样本选择方法
        use_mask_based_selection = (ablate == "only_mmd")
        use_gradient_matching = (ablate == "only_gradmatch") or (ablate not in ["only_mmd"])
        
        # 标准消融：使用梯度匹配方法选择核心集（GCR/CRAIG方法），消融时可为随机选择
        coreset_method = 'random' if ablate == 'no_gradient_matching' else 'greedy'
        if ablate == 'no_gradient_matching' and batch_count == 1:
            print("[Ablation] w/o gradient-matching selection: using random coreset.")
        if ablate == 'no_strategic_forgetting' and batch_count == 1:
            print("[Ablation] w/o strategic forgetting: fixed base_ratio/min_new_samples, ignoring drift for buffer update.")
        if ablate == 'no_regularization' and batch_count == 1:
            print("[Ablation] w/o regularization: no LwF when no drift (always train as drift branch, no knowledge distillation).")
        if ablate == "only_mmd" and batch_count == 1:
            print("[Ablation] only_mmd: using MMD drift detection with original mask-based sample selection.")
        
        # 根据消融实验类型选择样本选择方法
        if use_mask_based_selection:
            # 消融：only_mmd — 使用MMD漂移检测 + 原始mask-based样本选择
            # 需要使用anomaly scores作为mask优化的输入
            if dataset == 'nsl':
                # 对于NSL，使用正常概率占比作为响应值
                control_res = buffer_normal_prob_ratio.cpu().numpy()
                treatment_res = new_batch_normal_prob_ratio.cpu().numpy()
            else:
                # 对于UNSW，使用logits作为响应值（归一化到[0,1]）
                control_res = torch.sigmoid(buffer_logits).cpu().numpy()
                treatment_res = torch.sigmoid(new_batch_logits).cpu().numpy()
            
            # 优化mask
            M_c = optimize_old_mask(control_res, treatment_res, device, 
                                   initialization='0.5-1', lr=1.0, steps=100)
            M_t = optimize_new_mask(control_res, treatment_res, M_c, device,
                                   initialization='0-0.5', lr=50.0, steps=100)
            
            # 使用mask-based方法选择样本
            if drift_detected:
                if dataset == 'nsl':
                    training_buffer_x, training_buffer_y, selected_indices_in_batch, new_sample_mask = \
                        select_and_update_representative_samples_mask_drift(
                            training_buffer_x, training_buffer_y,
                            current_batch_x, current_batch_y,
                            M_c, M_t, num_labeled_sample, device, memory_buffer_size,
                            model, normal_reconstruction_template
                        )
                else:
                    training_buffer_x, training_buffer_y, selected_indices_in_batch, new_sample_mask = \
                        select_and_update_representative_samples_mask_drift(
                            training_buffer_x, training_buffer_y,
                            current_batch_x, current_batch_y,
                            M_c, M_t, num_labeled_sample, device, memory_buffer_size,
                            model, None
                        )
            else:
                training_buffer_x, training_buffer_y, selected_indices_in_batch, new_sample_mask = \
                    select_and_update_representative_samples_mask(
                        training_buffer_x, training_buffer_y,
                        current_batch_x, current_batch_y,
                        M_c, M_t, num_labeled_sample, device, memory_buffer_size
                    )
            
            # 转换为tensor（如果返回的是numpy array）
            if not isinstance(selected_indices_in_batch, torch.Tensor):
                selected_indices_in_batch = torch.tensor(selected_indices_in_batch, device=device)
            if not isinstance(new_sample_mask, torch.Tensor):
                new_sample_mask = torch.tensor(new_sample_mask, device=device)
        else:
            # 标准流程：使用梯度匹配核心集选择
            print("使用梯度匹配核心集选择..." if coreset_method == 'greedy' else "使用随机核心集选择...")
            training_buffer_x, training_buffer_y, selected_indices_in_batch, new_sample_mask = select_coreset_gradient_matching(
                training_buffer_x, training_buffer_y, current_batch_x, current_batch_y,
                memory_buffer_size, model, contrastive_loss_fn, device, dataset=dataset, 
                num_new_samples=num_labeled_sample, max_candidates=max_candidates,
                drift_detected=drift_detected, drift_score=drift_metric_value,
                drift_threshold=drift_threshold,
                old_scores=buffer_anomaly_scores,
                new_scores=new_batch_anomaly_scores,
                coreset_method=coreset_method,
                no_strategic_forgetting=(ablate == 'no_strategic_forgetting')
            )
        
        # 打印训练缓冲区的类别分布（用于调试）
        if dataset != 'nsl':
            unique_labels, label_counts = torch.unique(training_buffer_y, return_counts=True)
            print(f"训练缓冲区类别分布: ", end="")
            for label, count in zip(unique_labels, label_counts):
                label_name = "正常" if label == 0 else "异常"
                print(f"{label_name}={count.item()}", end=", ")
            print()
        
        # 将批次内索引转换为流式数据池中的全局索引
        global_selected_indices = start_idx - sample_interval + selected_indices_in_batch.cpu().numpy()
        all_selected_indices.append(global_selected_indices)

        # 使用选中的核心集重新训练模型
        coreset_dataset = TensorDataset(training_buffer_x, training_buffer_y, new_sample_mask)
        coreset_loader = DataLoader(
            dataset=coreset_dataset, batch_size=batch_size, shuffle=True
        )
        
        model = model.to(device)
        model.train()
        
        # 根据是否检测到漂移采用不同的训练策略（消融四：无漂移时也不用 LwF，走有漂移分支）
        use_drift_branch = drift_detected or (ablate == 'no_regularization')
        if use_drift_branch:
            # 检测到漂移（或消融 no_regularization）：直接训练，不使用知识蒸馏
            for epoch in range(epoch_1):
                if epoch % 50 == 0:
                    print(f'训练轮次 = {epoch}')
                for batch_data in coreset_loader:
                    inputs, labels, sample_mask = batch_data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    sample_mask = sample_mask.to(device)
                    normal_new_sample_mask = sample_mask[labels == 0]  # 新样本中正常样本的掩码
                    
                    optimizer.zero_grad()
                    
                    if dataset == 'nsl':
                        features, recon_vec = model(inputs)
                    else:
                        features, recon_vec, classifications = model(inputs)
                    
                    # 计算对比损失
                    contrastive_loss = contrastive_loss_fn(recon_vec, labels)
                    # 对新正常样本应用更高的权重
                    weighted_contrastive_loss = contrastive_loss * (
                        (1 - normal_new_sample_mask) + normal_new_sample_mask * new_sample_weight
                    )

                    if dataset == 'nsl':
                        total_loss = weighted_contrastive_loss.mean()
                    else:
                        classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                        # 应用类别权重
                        if class_weights is not None:
                            sample_weights = class_weights[labels.long()]
                            classification_loss = classification_loss * sample_weights
                        weighted_classification_loss = classification_loss * (
                            (1 - sample_mask) + sample_mask * new_sample_weight
                        )
                        total_loss = weighted_contrastive_loss.mean() + weighted_classification_loss.mean()

                    total_loss.backward()
                    optimizer.step()
        else:
            # 未检测到漂移：使用知识蒸馏防止灾难性遗忘
            for epoch in range(epoch_1):
                if epoch % 50 == 0:
                    print(f'训练轮次 = {epoch}')
                for batch_data in coreset_loader:
                    inputs, labels, sample_mask = batch_data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    sample_mask = sample_mask.to(device)
                    normal_new_sample_mask = sample_mask[labels == 0]
                    
                    optimizer.zero_grad()
                    
                    if dataset == 'nsl':
                        features, recon_vec = model(inputs)
                    else:
                        features, recon_vec, classifications = model(inputs)

                    # 计算对比损失
                    contrastive_loss = contrastive_loss_fn(recon_vec, labels)
                    weighted_contrastive_loss = contrastive_loss * (
                        (1 - normal_new_sample_mask) + normal_new_sample_mask * new_sample_weight
                    )

                    if dataset == 'nsl':
                        weighted_loss = weighted_contrastive_loss.mean()
                    else:
                        classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                        # 应用类别权重
                        if class_weights is not None:
                            sample_weights = class_weights[labels.long()]
                            classification_loss = classification_loss * sample_weights
                        weighted_classification_loss = classification_loss * (
                            (1 - sample_mask) + sample_mask * new_sample_weight
                        )
                        weighted_loss = weighted_contrastive_loss.mean() + weighted_classification_loss.mean()

                    # 知识蒸馏损失（LwF方法）防止灾难性遗忘
                    if dataset == 'nsl':
                        with torch.no_grad():
                            teacher_features, teacher_recon_vec = teacher_model(inputs)
                        distillation_loss = F.mse_loss(recon_vec, teacher_recon_vec)
                    else:
                        with torch.no_grad():
                            teacher_features, teacher_recon_vec, teacher_logits = teacher_model(inputs)
                        distillation_loss = F.mse_loss(classifications, teacher_logits)
                    
                    total_loss = weighted_loss + knowledge_distillation_weight * distillation_loss

                    total_loss.backward()
                    optimizer.step()
        
        # 更新教师模型（将当前学生模型参数复制到教师模型）
        teacher_model.load_state_dict(model.state_dict())
        
        # 评估当前批次的模型性能并存储预测结果
        if dataset == 'nsl':
            buffer_normal_mask = (training_buffer_y == 0).squeeze()
            normal_reconstruction_template = torch.mean(
                F.normalize(model(training_buffer_x[buffer_normal_mask])[1], p=2, dim=1), dim=0
            )
            batch_predictions = evaluate(
                normal_reconstruction_template, training_buffer_x, training_buffer_y,
                current_batch_x, 0, model
            )
        else:
            batch_dataset = TensorDataset(current_batch_x, current_batch_y)
            batch_loader = DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=False)
            # 使用优化后的阈值获取预测结果
            batch_predictions = evaluate_classifier(
                model, batch_loader, device, get_predict=True, optimize_threshold=True
            )
        
        all_batch_predictions = torch.cat((
            all_batch_predictions.to(device), 
            torch.tensor(batch_predictions).to(device)
        ))

        # 可选：每个批次后在固定测试集上评估 F1（用于真实持续学习曲线）
        if save_curve_csv and log_curve_per_batch:
            if dataset == 'nsl':
                # 使用当前模型与当前缓冲区模板在完整测试集上评估
                with torch.no_grad():
                    buffer_normal_mask = (training_buffer_y == 0).squeeze()
                    normal_reconstruction_template = torch.mean(
                        F.normalize(model(training_buffer_x[buffer_normal_mask])[1], p=2, dim=1), dim=0
                    )
                    full_test_pred = evaluate(
                        normal_reconstruction_template, training_buffer_x, training_buffer_y,
                        x_test, 0, model
                    )
                y_true_np = y_test.detach().cpu().numpy()
                y_pred_np = full_test_pred.detach().cpu().numpy().astype("int32")
                batch_f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0)) * 100.0
            else:
                # UNSW: 用优化阈值的预测结果评估完整测试集 F1
                full_test_dataset = TensorDataset(x_test, y_test)
                full_test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)
                full_test_pred = evaluate_classifier(
                    model, full_test_loader, device, get_predict=True, optimize_threshold=True
                )
                y_true_np = y_test.detach().cpu().numpy()
                batch_f1 = float(f1_score(y_true_np, full_test_pred, zero_division=0)) * 100.0

            # step 用 batch_count（从 1 开始）
            seed_curve.append((batch_count, batch_f1))

################### 持续学习后的最终性能评估 ###################

    test_set_size = len(x_test)  # 独立测试集的大小
    stream_data_pool_size = len(stream_data_pool_x)  # 流式数据池的总大小

    # 合并所有批次中被选中的样本索引
    all_selected_global_indices = np.hstack(all_selected_indices)

    # 创建掩码：True表示未标记样本，False表示已用于训练的样本
    unlabeled_mask = np.ones(stream_data_pool_size, dtype=bool)
    unlabeled_mask[all_selected_global_indices] = False

    # 提取测试集部分的掩码
    unlabeled_test_mask = unlabeled_mask[-test_set_size:]

    # 最终评估
    if dataset != 'nsl':
        # 使用最终模型重新评估测试集（确保使用最优阈值和最终模型状态）
        final_test_x = stream_data_pool_x[-test_set_size:][unlabeled_test_mask]
        final_test_y = stream_data_pool_y[-test_set_size:][unlabeled_test_mask]
        
        # 检查未标记测试集的类别分布
        if len(final_test_y) > 0:
            unique_labels, label_counts = torch.unique(final_test_y, return_counts=True)
            n_abnormal = label_counts[unique_labels == 1].sum().item() if len(unique_labels) > 1 and (unique_labels == 1).any() else 0
            print(f"测试集类别分布（未标记样本）: ", end="")
            for label, count in zip(unique_labels, label_counts):
                label_name = "正常" if label == 0 else "异常"
                print(f"{label_name}={count.item()}", end=", ")
            print(f"(总计: {len(final_test_y)})")
            
            # 如果未标记测试集中没有异常样本，使用完整测试集
            if n_abnormal == 0:
                print("警告: 未标记测试集中没有异常样本。使用完整测试集进行评估。")
                final_test_x = stream_data_pool_x[-test_set_size:]
                final_test_y = stream_data_pool_y[-test_set_size:]
                unique_labels_full, counts_full = torch.unique(final_test_y, return_counts=True)
                print(f"完整测试集类别分布: ", end="")
                for label, count in zip(unique_labels_full, counts_full):
                    label_name = "正常" if label == 0 else "异常"
                    print(f"{label_name}={count.item()}", end=", ")
                print(f"(总计: {len(final_test_y)})")
        
        final_test_dataset = TensorDataset(final_test_x, final_test_y)
        final_test_loader = DataLoader(dataset=final_test_dataset, batch_size=batch_size, shuffle=False)
        
        print('--------------------------- 持续学习后性能评估 -----------------------------------')
        final_performance = evaluate_classifier(model, final_test_loader, device, get_predict=False)
    else:
        # NSL数据集使用存储的预测结果
        test_pseudo_labels = all_batch_predictions[-test_set_size:][unlabeled_test_mask].to(device)
        test_true_labels = stream_data_pool_y[-test_set_size:][unlabeled_test_mask].to(device)
        
        print('--------------------------- 持续学习后性能评估 -----------------------------------')
        final_performance = score_detail(
            test_true_labels.cpu().numpy(), 
            test_pseudo_labels.cpu().numpy()
        )

    # 若不记录逐批次曲线，至少记录前后两点
    if save_curve_csv and not log_curve_per_batch:
        final_f1 = float(final_performance[3]) * 100.0
        seed_curve.append((batch_count, final_f1))

    if save_curve_csv:
        all_seed_curves.append(seed_curve)
        print(f"[Curve] seed {current_seed}: recorded {len(seed_curve)} points.")

# 5-seed 曲线汇总并输出 CSV
if save_curve_csv and len(all_seed_curves) > 0:
    # 仅保留各 seed 共有的前 min_len 个 step，确保可对齐平均
    min_len = min(len(c) for c in all_seed_curves)
    truncated = [c[:min_len] for c in all_seed_curves]

    # 自动命名输出文件
    if not curve_csv_path:
        tag = "full" if ablate == "none" else ablate
        curve_csv_path = f"data/continual_f1_curve_{dataset}_{tag}.csv"

    out_dir = os.path.dirname(curve_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 默认写为 plot_results.py 可直接读取的列名；额外提供标准差列
    with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "f1_ours", "f1_std"])
        for idx in range(min_len):
            step = truncated[0][idx][0]
            values = [curve[idx][1] for curve in truncated]
            writer.writerow([step, float(np.mean(values)), float(np.std(values))])

    print(f"[Curve] saved 5-seed aggregated curve to: {curve_csv_path}")


