"""
AutoPromptOptimizer: 基于错误诊断和 LLM 交互的自动 Prompt 优化器
使用训练好的 CLIP 模型进行零样本分类，并通过 LLM 迭代优化类别描述
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

# 导入 CLIP 模型
from models.clip import CLIPModel

# 导入 OpenAI SDK (用于 DeepSeek API)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. Please install: pip install openai")

# 使用硅基流动（SiliconFlow）的API
DEEPSEEK_API_KEY = "sk-mqubfpfslyohpdbryxjsnrntckfdizhhwrgsviwdisyabccq"
DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3.2"

class AutoPromptOptimizer:
    """
    自动 Prompt 优化器
    
    核心流程：
    1. 使用 LLM 生成初始专家知识 Prompts
    2. 在验证集上评估当前 Prompts 的性能
    3. 分析混淆矩阵，找到最严重的混淆类别对
    4. 使用 LLM 针对性地优化被混淆类别的描述
    5. A/B 测试新描述，仅接受性能提升的修改（贪婪策略）
    6. 记录历史，避免 LLM 重复无效尝试
    """
    
    def __init__(
        self, 
        class_names, 
        val_images, 
        val_labels,
        model_path=None,
        checkpoint_path=None,
        image_encoder_name='resnet50',
        text_encoder_name='bert-base-chinese',
        embed_dim=512,
        temperature=0.07,
        img_size=224,
        device='cuda:0',
        llm_api_func=None,
        initial_prompts_file=None
    ):
        """
        Args:
            class_names: 类别名称列表，例如 ['Normal', 'Loosening', ...]
            val_images: 验证集图像路径列表或 PIL Image 对象列表
            val_labels: 验证集真实标签（整数索引列表）
            model_path: 已训练模型的路径（.pth checkpoint 文件）
            checkpoint_path: 同 model_path（兼容性参数）
            image_encoder_name: 图像编码器名称（如果从 checkpoint 加载，会从 checkpoint 读取）
            text_encoder_name: 文本编码器名称（如果从 checkpoint 加载，会从 checkpoint 读取）
            embed_dim: 嵌入维度（如果从 checkpoint 加载，会从 checkpoint 读取）
            temperature: 温度参数（如果从 checkpoint 加载，会从 checkpoint 读取）
            img_size: 图像大小
            device: 设备（'cuda:0' 或 'cpu'）
            llm_api_func: LLM API 调用函数，签名: func(system=None, user=None) -> str
            initial_prompts_file: 初始 prompts JSON 文件路径（如果提供，将从文件加载而不是使用 LLM 生成）
        """
        self.class_names = class_names
        self.val_images = val_images
        self.val_labels = np.array(val_labels) if not isinstance(val_labels, np.ndarray) else val_labels
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.llm_api_func = llm_api_func or self._default_llm_api
        
        # 初始化 DeepSeek API（如果使用默认 LLM API）
        self.deepseek_client = None
        self.deepseek_messages = []  # 维护对话历史（参考 OpenAI 的消息拼接方式）
        if self.llm_api_func == self._default_llm_api and OPENAI_AVAILABLE:
            try:
                self.deepseek_client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL
                )
                print("✓ DeepSeek API 初始化成功")
            except Exception as e:
                print(f"Warning: DeepSeek API 初始化失败: {e}")
                self.deepseek_client = None
        
        # 图像预处理（验证时不需要数据增强）
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载 CLIP 模型
        self.model = self._load_model(
            model_path or checkpoint_path,
            image_encoder_name,
            text_encoder_name,
            embed_dim,
            temperature
        )
        self.model.eval()
        self.model.to(self.device)
        
        # 初始化 Prompts（从文件加载或使用 LLM 生成）
        if initial_prompts_file and os.path.exists(initial_prompts_file):
            print(f"从文件加载初始 Prompts: {initial_prompts_file}")
            self.current_prompts = self.load_prompts_from_file(initial_prompts_file)
        else:
            print("正在使用 LLM 生成初始专家知识 Prompts...")
            self.current_prompts = self.initialize_prompts_with_llm()
        
        # 记录历史，用于给 LLM 提供上下文
        self.history = []
        self.best_score = 0.0
        self.best_prompts = None
        
    def _load_model(self, checkpoint_path, image_encoder_name, text_encoder_name, embed_dim, temperature):
        """加载 CLIP 模型（从 checkpoint 或创建新模型）"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"从 checkpoint 加载模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 从 checkpoint 读取配置（如果存在）
            if 'config' in checkpoint:
                config = checkpoint['config']
                image_encoder_name = config.get('image_encoder', image_encoder_name)
                text_encoder_name = config.get('text_encoder', text_encoder_name)
                embed_dim = config.get('embed_dim', embed_dim)
                temperature = config.get('temperature', temperature)
            
            # 创建模型
            model = CLIPModel(
                image_encoder_name=image_encoder_name,
                text_encoder_name=text_encoder_name,
                embed_dim=embed_dim,
                temperature=temperature
            )
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✓ 模型加载成功 (image_encoder: {image_encoder_name}, text_encoder: {text_encoder_name})")
        else:
            print(f"创建新模型 (image_encoder: {image_encoder_name}, text_encoder: {text_encoder_name})")
            model = CLIPModel(
                image_encoder_name=image_encoder_name,
                text_encoder_name=text_encoder_name,
                embed_dim=embed_dim,
                temperature=temperature
            )
        
        return model
    
    def _default_llm_api(self, system=None, user=None, use_stream=False):
        """
        默认 LLM API 调用函数（使用硅基流动的 DeepSeek-V3 API）
        参考 OpenAI 的消息拼接方式，维护对话历史
        
        Args:
            system: 系统提示词
            user: 用户消息
            use_stream: 是否使用流式输出（默认False）
        """
        if user is None:
            return "Default prompt description"
        
        # 如果 DeepSeek 不可用，返回占位符
        if not OPENAI_AVAILABLE or self.deepseek_client is None:
            print(f"[LLM API] System: {system}")
            print(f"[LLM API] User: {user[:100]}...")
            return f"Generated description for: {user[:50]}"
        
        try:
            # 构建消息（参考 OpenAI 的消息拼接方式）
            # 如果是第一次调用且有 system 提示，添加 system 消息
            if system and len(self.deepseek_messages) == 0:
                self.deepseek_messages.append({
                    "role": "system",
                    "content": system
                })
            
            # 添加用户消息到历史
            self.deepseek_messages.append({
                "role": "user",
                "content": user
            })
            
            # 调用 DeepSeek API（使用硅基流动）
            response = self.deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=self.deepseek_messages,
                temperature=0.7,
                max_tokens=1024,
                stream=use_stream
            )
            
            # 处理响应（流式或非流式）
            if use_stream:
                # 流式输出
                response_text = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        response_text += delta.content
                        print(delta.content, end="", flush=True)
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        response_text += delta.reasoning_content
                        print(delta.reasoning_content, end="", flush=True)
                print()  # 换行
                response_text = response_text.strip()
            else:
                # 非流式输出
                response_text = response.choices[0].message.content.strip()
            
            # 添加助手响应到历史（参考 OpenAI 的消息拼接方式）
            # 对于流式输出，需要手动构建消息对象
            if use_stream:
                self.deepseek_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
            else:
                self.deepseek_messages.append(response.choices[0].message)
            
            return response_text
            
        except Exception as e:
            error_msg = str(e)
            # 提供更友好的错误信息
            if "402" in error_msg or "Insufficient Balance" in error_msg or "余额" in error_msg:
                print(f"❌ API调用失败: 账户余额不足，请检查硅基流动账户余额")
            elif "401" in error_msg or "Unauthorized" in error_msg or "认证" in error_msg:
                print(f"❌ API调用失败: API密钥无效或已过期，请检查API密钥")
            elif "429" in error_msg or "Rate limit" in error_msg or "限流" in error_msg:
                print(f"❌ API调用失败: 请求频率过高，请稍后重试")
            else:
                print(f"❌ API调用失败: {error_msg}")
            
            # 如果出错，从历史中移除刚才添加的用户消息
            if self.deepseek_messages and self.deepseek_messages[-1]["role"] == "user":
                self.deepseek_messages.pop()
            return f"Error: {error_msg}"
    
    def _load_image(self, image_path_or_obj):
        """加载图像（支持路径字符串或 PIL Image 对象）"""
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj).convert('RGB')
        elif isinstance(image_path_or_obj, Image.Image):
            image = image_path_or_obj.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_path_or_obj)}")
        return image
    
    def initialize_prompts_with_llm(self):
        """
        Step 0: 让 LLM 基于医学知识生成初始详细描述
        
        Returns:
            prompts: 字典，{class_name: description}
        """
        prompts = {}
        
        for cls in tqdm(self.class_names, desc="生成初始 Prompts"):
            prompt = self.llm_api_func(
                system="你是骨科影像专家。请用英文生成详细、准确的医学影像描述。",
                user=f"请生成一段用于 CLIP 模型的详细英文描述，描述髋关节术后 X 光片中 '{cls}' 类别的视觉特征。"
                     f"要求：1) 准确描述该类别的医学特征；2) 控制在 50 个词以内；3) 使用专业但易懂的术语。"
            )
            prompts[cls] = prompt.strip()
            print(f"  {cls}: {prompts[cls][:80]}...")
        
        return prompts
    
    def load_prompts_from_file(self, filepath):
        """
        从 JSON 文件加载初始 Prompts（用于初始化，不更新历史记录）
        
        Args:
            filepath: JSON 文件路径，格式应包含 'prompts' 键
        
        Returns:
            prompts: 字典，{class_name: description}
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查文件格式
        if 'prompts' not in data:
            raise ValueError(f"JSON文件格式错误: 缺少 'prompts' 键。文件应包含 'prompts' 字典。")
        
        prompts = data['prompts']
        
        # 验证所有类别都有对应的 prompt
        missing_classes = set(self.class_names) - set(prompts.keys())
        if missing_classes:
            print(f"⚠️  警告: 以下类别在文件中缺少 prompts: {missing_classes}")
            print(f"   将使用 LLM 为这些类别生成 prompts...")
            # 为缺失的类别生成 prompts
            for cls in missing_classes:
                prompt = self.llm_api_func(
                    system="你是骨科影像专家。请用英文生成详细、准确的医学影像描述。",
                    user=f"请生成一段用于 CLIP 模型的详细英文描述，描述髋关节术后 X 光片中 '{cls}' 类别的视觉特征。"
                         f"要求：1) 准确描述该类别的医学特征；2) 控制在 50 个词以内；3) 使用专业但易懂的术语。"
                )
                prompts[cls] = prompt.strip()
                print(f"  {cls}: {prompts[cls][:80]}...")
        
        # 检查是否有多余的类别（不在 class_names 中）
        extra_classes = set(prompts.keys()) - set(self.class_names)
        if extra_classes:
            print(f"⚠️  警告: 文件中有以下额外的类别（将被忽略）: {extra_classes}")
            # 只保留 class_names 中的类别
            prompts = {cls: prompts[cls] for cls in self.class_names if cls in prompts}
        
        print(f"✓ 成功加载 {len(prompts)} 个类别的 Prompts")
        for cls in self.class_names:
            if cls in prompts:
                print(f"  {cls}: {prompts[cls][:80]}...")
        
        return prompts
    
    def  evaluate(self, prompts_dict):
        """
        运行 CLIP 进行 Zero-Shot 预测，返回准确率和混淆矩阵
        
        Args:
            prompts_dict: 类别描述字典，{class_name: description}
        
        Returns:
            accuracy: 准确率 (0-1)
            confusion_matrix: 混淆矩阵 [num_classes, num_classes]
        """
        # 将 prompts 字典转换为列表顺序（与 class_names 对应）
        text_inputs = [prompts_dict[cls] for cls in self.class_names]
        
        # 批量处理图像
        batch_size = 32
        all_predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.val_images), batch_size), desc="评估中"):
                batch_images = self.val_images[i:i+batch_size]
                
                # 加载和预处理图像
                image_tensors = []
                for img in batch_images:
                    image = self._load_image(img)
                    image_tensor = self.transform(image)
                    image_tensors.append(image_tensor)
                
                image_tensors = torch.stack(image_tensors).to(self.device)
                
                # 使用 CLIP 模型预测
                predictions, _ = self.model.predict(image_tensors, class_texts=text_inputs)
                all_predictions.extend(predictions.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        
        # 计算指标
        acc = accuracy_score(self.val_labels, all_predictions)
        cm = confusion_matrix(
            self.val_labels, 
            all_predictions, 
            labels=range(len(self.class_names))
        )
        
        return acc, cm
    
    def find_worst_confusion(self, cm):
        """
        分析混淆矩阵，找到混淆最严重的两个类别
        
        Args:
            cm: 混淆矩阵 [num_classes, num_classes]
        
        Returns:
            actual_class: 真实类别名称
            predicted_class: 被误判为的类别名称
            error_count: 错误次数
        """
        # 将对角线置为 0，因为我们只关心错误
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        
        # 找到最大错误值的索引
        max_error_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        actual_idx, predicted_idx = max_error_idx
        
        actual_class = self.class_names[actual_idx]
        predicted_class = self.class_names[predicted_idx]
        error_count = int(cm_copy[actual_idx, predicted_idx])
        
        return actual_class, predicted_class, error_count
    
    def get_all_confusions_sorted(self, cm, top_k=None):
        """
        获取所有混淆对，按严重程度排序
        
        Args:
            cm: 混淆矩阵 [num_classes, num_classes]
            top_k: 返回前k个混淆对（如果为None，返回所有非零混淆）
        
        Returns:
            confusions: 列表，每个元素为 (actual_class, predicted_class, error_count)
                       按 error_count 降序排列
        """
        # 将对角线置为 0，因为我们只关心错误
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        
        # 获取所有非零混淆
        confusions = []
        for actual_idx in range(len(self.class_names)):
            for predicted_idx in range(len(self.class_names)):
                error_count = int(cm_copy[actual_idx, predicted_idx])
                if error_count > 0:
                    actual_class = self.class_names[actual_idx]
                    predicted_class = self.class_names[predicted_idx]
                    confusions.append((actual_class, predicted_class, error_count))
        
        # 按错误次数降序排序
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        # 如果指定了top_k，只返回前k个
        if top_k is not None:
            confusions = confusions[:top_k]
        
        return confusions
    
    def get_history_context(self, cls_name):
        """
        获取某个类别之前的失败尝试，防止 LLM 生成重复的无效 Prompt
        
        Args:
            cls_name: 类别名称
        
        Returns:
            context_str: 历史上下文字符串
        """
        failures = []
        for h in self.history:
            if h.get('class') == cls_name and h.get('result') == 'failed':
                # 兼容 'desc' 和 'new_desc' 两种键名
                desc = h.get('new_desc') or h.get('desc', '')
                if desc:  # 只添加非空的描述
                    failures.append(desc)
        
        if not failures:
            return "无历史失败记录。"
        
        # 只返回最近 3 次失败尝试
        recent_failures = failures[-3:]
        context = "之前的这些描述尝试过但效果不好，请避免类似写法：\n"
        for i, desc in enumerate(recent_failures, 1):
            context += f"{i}. {desc}\n"
        
        return context
    
    def optimize_loop(self, max_rounds=10, min_improvement=0.001, patience=10):
        """
        核心闭环流程：迭代优化 Prompts
        
        Args:
            max_rounds: 已弃用，不再使用（保留参数以保持向后兼容）
            min_improvement: 最小改进阈值（低于此值认为无提升）
            patience: 早停耐心值（连续 N 轮无提升则切换到下一个混淆对）
        
        Returns:
            best_prompts: 最佳 Prompts 字典
        
        注意：优化器将持续运行直到所有混淆对都尝试过（每个混淆对连续patience轮无提升后切换）
        """
        # 初始评估
        print("\n" + "="*80)
        print("开始优化流程")
        print("="*80)
        
        current_acc, current_cm = self.evaluate(self.current_prompts)
        self.best_score = current_acc
        self.best_prompts = self.current_prompts.copy()
        
        print(f"\n初始准确率: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        self._print_confusion_summary(current_cm)
        
        # 获取所有混淆对（按严重程度排序）
        all_confusions = self.get_all_confusions_sorted(current_cm)
        if len(all_confusions) == 0:
            print("✓ 没有发现混淆错误，优化完成！")
            return self.best_prompts
        
        print(f"\n发现 {len(all_confusions)} 个混淆对，将按严重程度依次优化")
        
        # 跟踪当前混淆对索引和连续无提升计数
        confusion_idx = 0
        no_improvement_count = 0  # 当前混淆对的连续无提升计数
        total_rounds = 0
        
        # 记录已"耗尽"的混淆对（连续patience轮无提升的混淆对）
        exhausted_confusions = set()
        
        # 循环条件：持续运行直到所有混淆对都尝试过
        while True:
            # 如果当前混淆对索引超出范围，说明所有混淆对都已尝试过
            if confusion_idx >= len(all_confusions):
                print(f"\n所有混淆对都已尝试，重新评估混淆矩阵...")
                current_acc, current_cm = self.evaluate(self.current_prompts)
                all_confusions = self.get_all_confusions_sorted(current_cm)
                
                if len(all_confusions) == 0:
                    print("✓ 没有发现混淆错误，优化完成！")
                    break
                
                # 过滤掉已耗尽的混淆对
                remaining_confusions = []
                for act, pred, cnt in all_confusions:
                    confusion_key = (act, pred)
                    if confusion_key not in exhausted_confusions:
                        remaining_confusions.append((act, pred, cnt))
                
                if len(remaining_confusions) == 0:
                    print(f"⚠️  所有混淆对都已尝试过 {patience} 次且无提升，停止优化")
                    break
                
                all_confusions = remaining_confusions
                print(f"发现 {len(all_confusions)} 个混淆对（已过滤掉已耗尽的混淆对），重新开始优化")
                confusion_idx = 0
                no_improvement_count = 0  # 重置计数
            
            # 获取当前要优化的混淆对
            act_cls, pred_cls, count = all_confusions[confusion_idx]
            confusion_key = (act_cls, pred_cls)
            
            # 如果这个混淆对已经耗尽，跳过它
            if confusion_key in exhausted_confusions:
                confusion_idx += 1
                continue
            
            total_rounds += 1
            print(f"\n{'='*80}")
            print(f"Round {total_rounds} (混淆对 {confusion_idx + 1}/{len(all_confusions)})")
            print(f"{'='*80}")
            print(f"当前错误: 真实类别 '{act_cls}' 被误判为 '{pred_cls}' (次数: {count})")
            print(f"连续无提升计数: {no_improvement_count}/{patience}")
            
            # 2. 构建 Prompt 给 LLM (包含上下文)
            current_desc = self.current_prompts[act_cls]
            confusing_desc = self.current_prompts[pred_cls]
            
            llm_prompt = f"""
                        你正在优化 CLIP 模型的分类能力。

                        【当前状态】
                        模型经常将类别 "{act_cls}" 误判为 "{pred_cls}"。

                        【当前描述】
                        "{act_cls}": {current_desc}
                        "{pred_cls}": {confusing_desc}

                        【任务】
                        请重写 "{act_cls}" 的描述。
                        要求：
                        1. 保留核心医学语义。
                        2. 重点强调它与 "{pred_cls}" 的视觉差异。
                        3. 不要太长，控制在 50 个词以内。
                        4. 使用清晰、具体的视觉特征描述。

                        【历史尝试失败记录】
                        {self.get_history_context(act_cls)}

                        请只返回新的描述文本，不要包含其他解释。
                        """
            
            # 3. LLM 生成新 Prompt
            print(f"\n正在请求 LLM 优化 '{act_cls}' 的描述...")
            new_desc_candidate = self.llm_api_func(
                system="你是骨科影像专家，擅长优化医学影像分类模型的文本描述。",
                user=llm_prompt
            ).strip()
            
            print(f"LLM 建议的新描述:\n  {new_desc_candidate}")
            
            # 4. 创建临时 Prompts 字典进行测试 (A/B Test)
            test_prompts = self.current_prompts.copy()
            test_prompts[act_cls] = new_desc_candidate
            
            # 5. 验证效果
            print(f"\n正在评估新描述的效果...")
            new_acc, new_cm = self.evaluate(test_prompts)
            
            # 6. 决策与更新 (贪婪策略)
            # 计算相对于历史最佳的改进
            improvement = new_acc - self.best_score
            # 计算相对于当前准确率的改进（用于显示）
            current_acc = self.best_score  # 当前使用的是历史最佳
            
            if improvement > min_improvement:
                print(f"\n✅ 优化成功! 准确率提升: {current_acc:.4f} -> {new_acc:.4f} "
                      f"(提升: +{improvement:.4f}, +{improvement*100:.2f}%)")
                
                self.best_score = new_acc
                self.current_prompts = test_prompts
                self.best_prompts = test_prompts.copy()
                current_cm = new_cm
                no_improvement_count = 0  # 重置连续无提升计数（因为成功了）
                
                # 重新获取混淆列表（因为混淆矩阵已更新）
                all_confusions = self.get_all_confusions_sorted(current_cm)
                if len(all_confusions) == 0:
                    print("✓ 没有发现混淆错误，优化完成！")
                    break
                
                # 重新定位当前混淆对的索引（可能已经变化或消失）
                confusion_idx = 0
                found = False
                for idx, (a, p, c) in enumerate(all_confusions):
                    if a == act_cls and p == pred_cls:
                        confusion_idx = idx
                        found = True
                        break
                
                # 如果当前混淆对已经解决（不在新列表中），从第一个开始
                if not found:
                    print(f"✓ 混淆对 '{act_cls}' -> '{pred_cls}' 已解决，从第一个混淆对重新开始")
                    confusion_idx = 0
                    no_improvement_count = 0  # 重置计数
                
                # 记录成功的修改
                self.history.append({
                    'round': total_rounds,
                    'class': act_cls,
                    'old_desc': current_desc,
                    'new_desc': new_desc_candidate,
                    'result': 'success',
                    'improvement': improvement
                })
            else:
                # 提供更详细的失败信息
                if abs(improvement) < 1e-6:  # 浮点数比较，几乎相等
                    print(f"\n❌ 优化失败: 新准确率 {new_acc:.4f} 等于历史最佳 {self.best_score:.4f}，无改进")
                elif improvement < 0:
                    print(f"\n❌ 优化失败: 新准确率 {new_acc:.4f} 低于历史最佳 {self.best_score:.4f} "
                          f"(下降: {improvement:.4f}, {improvement*100:.2f}%)")
                else:
                    print(f"\n❌ 优化失败: 新准确率 {new_acc:.4f} 提升 {improvement:.4f} "
                          f"但未达到最小改进阈值 {min_improvement:.4f}")
                print(f"   保留旧描述。")
                
                no_improvement_count += 1
                
                # 记录失败，避免 LLM 重蹈覆辙
                self.history.append({
                    'round': total_rounds,
                    'class': act_cls,
                    'old_desc': current_desc,
                    'new_desc': new_desc_candidate,
                    'result': 'failed',
                    'improvement': improvement
                })
                
                # 如果连续patience轮无提升，标记当前混淆对为已耗尽，切换到下一个混淆对
                if no_improvement_count >= patience:
                    print(f"\n⚠️  连续 {patience} 轮无提升，标记混淆对 '{act_cls}' -> '{pred_cls}' 为已耗尽，切换到下一个混淆对")
                    exhausted_confusions.add(confusion_key)
                    confusion_idx += 1
                    no_improvement_count = 0  # 重置计数，因为切换了混淆对
                    
                    # 如果所有混淆对都已尝试过，会在下次循环时重新评估混淆矩阵
                    if confusion_idx >= len(all_confusions):
                        continue  # 继续循环，会在下次迭代时重新评估混淆矩阵
        
        print(f"\n{'='*80}")
        print("优化结束")
        print(f"{'='*80}")
        print(f"最终准确率: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        print(f"总优化轮数: {len(self.history)}")
        print(f"成功优化次数: {sum(1 for h in self.history if h['result'] == 'success')}")
        
        return self.best_prompts
    
    def _print_confusion_summary(self, cm):
        """打印混淆矩阵摘要"""
        print("\n混淆矩阵摘要（前 5 个最严重的错误）:")
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        
        # 找到前 5 个最大的错误
        flat_indices = np.argsort(cm_copy.flatten())[::-1]
        top_errors = []
        for idx in flat_indices:
            if cm_copy.flatten()[idx] > 0:
                actual_idx, pred_idx = np.unravel_index(idx, cm_copy.shape)
                top_errors.append((actual_idx, pred_idx, int(cm_copy[actual_idx, pred_idx])))
                if len(top_errors) >= 5:
                    break
        
        for i, (act_idx, pred_idx, count) in enumerate(top_errors, 1):
            print(f"  {i}. {self.class_names[act_idx]} -> {self.class_names[pred_idx]}: {count} 次")
    
    def save_prompts(self, filepath, prompts=None):
        """保存优化后的 Prompts 到 JSON 文件"""
        if prompts is None:
            prompts = self.best_prompts or self.current_prompts
        
        output = {
            'class_names': self.class_names,
            'prompts': prompts,
            'best_accuracy': float(self.best_score),
            'optimization_history': self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Prompts 已保存到: {filepath}")
    
    def load_prompts(self, filepath):
        """从 JSON 文件加载 Prompts"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.current_prompts = data['prompts']
        if 'best_accuracy' in data:
            self.best_score = data['best_accuracy']
        if 'optimization_history' in data:
            self.history = data['optimization_history']
        
        print(f"✓ Prompts 已从 {filepath} 加载")
        return self.current_prompts


# ==================== 辅助函数 ====================

def load_validation_set_from_fold(train_output_dir, fold_num, data_dir, random_state=42, n_splits=None):
    """
    从交叉验证的fold中加载验证集（与训练时使用的验证集一致）
    
    Args:
        train_output_dir: 训练输出目录（包含fold_N子目录）
        fold_num: fold编号（1-based）
        data_dir: 原始数据目录（按类别组织的文件夹）
        random_state: 随机种子（必须与训练时一致）
        n_splits: 交叉验证折数（如果为None，则从config.json读取或使用默认值5）
    
    Returns:
        val_images: 验证集图像路径列表
        val_labels: 验证集标签列表
        class_names: 类别名称列表
    """
    try:
        from train_clip import CLIPDataset, create_folds_from_dataset
    except ImportError:
        # 尝试相对导入
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from train_clip import CLIPDataset, create_folds_from_dataset
        except ImportError as e:
            raise ImportError(f"无法导入train_clip模块: {e}。请确保train_clip.py在同一目录下。")
    
    # 尝试从配置文件读取n_splits和random_state
    config_path = Path(train_output_dir) / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if n_splits is None:
                    n_splits = config.get('n_splits', 5)
                if random_state == 42:  # 如果使用默认值，尝试从配置读取
                    random_state = config.get('random_state', 42)
                print(f"✓ 从配置文件读取: n_splits={n_splits}, random_state={random_state}")
        except Exception as e:
            print(f"⚠️  读取配置文件失败: {e}，使用默认值")
    
    if n_splits is None:
        n_splits = 5
    
    # 加载完整数据集
    full_dataset = CLIPDataset(data_dir, transform=None)
    class_names = sorted(full_dataset.class_to_idx.keys())
    
    # 创建folds（使用与训练时相同的随机种子）
    folds = create_folds_from_dataset(full_dataset, n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 获取指定fold的验证集索引
    if fold_num < 1 or fold_num > len(folds):
        raise ValueError(f"Fold编号必须在1-{len(folds)}之间，当前为{fold_num}")
    
    train_indices, val_indices = folds[fold_num - 1]  # fold_num是1-based
    
    # 从验证集索引中提取图像路径和标签
    val_images = []
    val_labels = []
    for idx in val_indices:
        img_path, label, _ = full_dataset.samples[idx]
        val_images.append(img_path)
        val_labels.append(label)
    
    print(f"✓ 从Fold {fold_num}加载验证集: {len(val_images)} 张图像")
    return val_images, val_labels, class_names


def load_validation_set_from_dir(data_dir, class_names=None):
    """
    从数据目录加载验证集（使用与train_clip.py相同的CLIPDataset结构）
    
    Args:
        data_dir: 数据目录（按类别组织的文件夹）
        class_names: 类别名称列表（如果为None，则从目录中自动获取）
    
    Returns:
        val_images: 验证集图像路径列表
        val_labels: 验证集标签列表
        class_names: 类别名称列表
    """
    try:
        from train_clip import CLIPDataset
    except ImportError:
        # 尝试相对导入
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from train_clip import CLIPDataset
        except ImportError as e:
            raise ImportError(f"无法导入train_clip模块: {e}。请确保train_clip.py在同一目录下。")
    
    # 加载数据集
    full_dataset = CLIPDataset(data_dir, transform=None)
    
    # 获取类别名称
    if class_names is None:
        class_names = sorted(full_dataset.class_to_idx.keys())
    else:
        # 验证类别名称是否匹配
        dataset_classes = set(full_dataset.class_to_idx.keys())
        input_classes = set(class_names)
        if dataset_classes != input_classes:
            print(f"⚠️  警告: 数据目录中的类别与提供的类别名称不完全匹配")
            print(f"  数据目录类别: {sorted(dataset_classes)}")
            print(f"  提供的类别: {sorted(input_classes)}")
            # 使用数据目录中的类别
            class_names = sorted(dataset_classes)
    
    # 提取所有图像路径和标签
    val_images = []
    val_labels = []
    for img_path, label, _ in full_dataset.samples:
        val_images.append(img_path)
        val_labels.append(label)
    
    print(f"✓ 从数据目录加载验证集: {len(val_images)} 张图像, {len(class_names)} 个类别")
    return val_images, val_labels, class_names


# ==================== 分步测试函数 ====================

def step_by_step_test():
    """
    分步测试各个功能模块
    用户可以逐步测试每个功能，确保每个模块正常工作
    """
    import glob
    from pathlib import Path
    
    print("="*80)
    print("AutoPromptOptimizer 分步测试")
    print("="*80)
    
    # ========== Step 1: 准备测试数据 ==========
    print("\n【Step 1】准备测试数据")
    print("-" * 80)
    
    # 配置路径（请根据实际情况修改）
    checkpoint_path = input("请输入模型 checkpoint 路径（留空使用默认）: ").strip()
    if not checkpoint_path:
        checkpoint_path = 'checkpoints/clip_models/resnet18_clip_ViT-B_32/fold_4/checkpoint_best.pth'
    
    # 尝试从checkpoint中读取类别信息
    class_names = None
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'class_to_idx' in checkpoint:
                # 从class_to_idx构建class_names（按索引排序）
                class_to_idx = checkpoint['class_to_idx']
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                print(f"✓ 从checkpoint读取类别信息: {len(class_names)} 个类别")
            elif 'class_texts' in checkpoint:
                # 如果有class_texts，尝试推断类别名称
                class_texts = checkpoint['class_texts']
                if isinstance(class_texts, list) and len(class_texts) > 0:
                    print(f"⚠️  Checkpoint中有class_texts但无class_to_idx，将使用默认类别名称")
        except Exception as e:
            print(f"⚠️  读取checkpoint失败: {e}")
    
    # 如果无法从checkpoint读取，使用默认类别名称
    if class_names is None:
        class_names = [
            'Acetabular Loosening', 'Dislocation', 'Fracture', 'Good Place', 
            'Infection', 'Native Hip', 'Spacer', 'Stem Loosening', 'Wear'
        ]
        print(f"使用默认类别名称: {len(class_names)} 个类别")
    
    print(f"类别列表: {class_names}")
    
    # 选择验证集加载方式
    print("\n验证集加载方式:")
    print("  1. 从交叉验证fold加载（与训练时使用的验证集一致）")
    print("  2. 从数据目录加载（使用所有数据作为验证集）")
    print("  3. 跳过图像测试")
    
    load_mode = input("请选择加载方式 (1/2/3, 默认3): ").strip()
    if not load_mode:
        load_mode = '3'
    
    val_images = []
    val_labels = []
    
    if load_mode == '1':
        # 从交叉验证fold加载
        print("\n说明：训练输出目录是运行 train_clip.py 时使用 --output-dir 参数指定的目录，")
        print("      该目录应包含 fold_1/, fold_2/, fold_3/ 等子目录，以及 config.json 文件。")
        print("      例如：如果训练时使用 --output-dir checkpoints/my_model，")
        print("           则应该输入: checkpoints/my_model")
        print()
        
        # 先尝试从checkpoint路径自动推断
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        inferred_output_dir = None
        inferred_fold_num = None
        
        if 'fold_' in checkpoint_dir:
            # 尝试推断训练输出目录（checkpoint的父目录的父目录）
            inferred_output_dir = os.path.dirname(checkpoint_dir)
            # 检查是否包含fold_N子目录
            if os.path.exists(inferred_output_dir):
                fold_dirs = [d for d in os.listdir(inferred_output_dir) 
                           if os.path.isdir(os.path.join(inferred_output_dir, d)) and d.startswith('fold_')]
                if fold_dirs:
                    print(f"✓ 自动检测到训练输出目录: {inferred_output_dir}")
                    print(f"  发现 {len(fold_dirs)} 个fold目录: {sorted(fold_dirs)}")
                    # 推断fold编号
                    fold_name = os.path.basename(checkpoint_dir)
                    if fold_name.startswith('fold_'):
                        try:
                            inferred_fold_num = int(fold_name.split('fold_')[-1])
                            print(f"✓ 自动检测到fold编号: {inferred_fold_num}")
                        except:
                            pass
        
        train_output_dir = input(f"请输入训练输出目录（留空使用自动检测: {inferred_output_dir or '无'}）: ").strip()
        if not train_output_dir:
            train_output_dir = inferred_output_dir
        
        if train_output_dir:
            # 验证目录是否存在且包含fold子目录
            if not os.path.exists(train_output_dir):
                print(f"❌ 目录不存在: {train_output_dir}")
                train_output_dir = None
            else:
                fold_dirs = [d for d in os.listdir(train_output_dir) 
                           if os.path.isdir(os.path.join(train_output_dir, d)) and d.startswith('fold_')]
                if not fold_dirs:
                    print(f"⚠️  警告: 目录 {train_output_dir} 中未找到 fold_N 子目录")
                    print("  该目录可能不是训练输出目录，请确认路径是否正确")
                else:
                    print(f"✓ 找到 {len(fold_dirs)} 个fold目录: {sorted(fold_dirs)}")
        
        if train_output_dir:
            fold_num_str = input(f"请输入fold编号（1-5，留空使用自动检测: {inferred_fold_num or '无'}）: ").strip()
            if not fold_num_str:
                fold_num = inferred_fold_num
                if fold_num is None:
                    # 如果无法推断，使用第一个fold
                    fold_num = 1
                    print(f"  无法推断fold编号，使用默认值: {fold_num}")
            else:
                fold_num = int(fold_num_str)
            
            default_data_dir = '/home/ln/wangweicheng/ModelsTotrain/single_label_data'
            data_dir = input(f"请输入原始数据目录（按类别组织的文件夹，与训练时使用的数据目录相同，留空使用默认: {default_data_dir}）: ").strip()
            if not data_dir:
                data_dir = default_data_dir
            if data_dir and os.path.exists(data_dir):
                try:
                    val_images, val_labels, class_names = load_validation_set_from_fold(
                        train_output_dir, fold_num, data_dir, random_state=42
                    )
                except Exception as e:
                    print(f"❌ 从fold加载验证集失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("  将尝试从数据目录加载...")
                    load_mode = '2'
            else:
                print("⚠️  数据目录不存在，将尝试从数据目录加载...")
                load_mode = '2'
        else:
            print("⚠️  无法确定训练输出目录，将尝试从数据目录加载...")
            load_mode = '2'
    
    if load_mode == '2':
        # 从数据目录加载
        default_data_dir = '/home/ln/wangweicheng/ModelsTotrain/single_label_data'
        data_dir = input(f"请输入验证集数据目录（留空使用默认: {default_data_dir}，或输入'skip'跳过图像测试）: ").strip()
        if not data_dir:
            data_dir = default_data_dir
        elif data_dir.lower() == 'skip':
            print("⚠️  跳过图像测试，将使用模拟数据进行部分测试")
            val_images = None
            val_labels = None
            data_dir = None
        else:
            # 用户输入了自定义路径
            pass
        
        if data_dir and os.path.exists(data_dir):
            try:
                val_images, val_labels, class_names = load_validation_set_from_dir(data_dir, class_names)
            except Exception as e:
                print(f"❌ 从数据目录加载验证集失败: {e}")
                import traceback
                traceback.print_exc()
                print("  将使用传统方式加载...")
                # 回退到传统方式
                val_images = []
                val_labels = []
                for class_idx, class_name in enumerate(class_names):
                    class_dir = os.path.join(data_dir, class_name)
                    if os.path.exists(class_dir):
                        images = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                                 glob.glob(os.path.join(class_dir, '*.png'))
                        val_images.extend(images)
                        val_labels.extend([class_idx] * len(images))
                        print(f"  {class_name}: {len(images)} 张图像")
            
            if len(val_images) == 0:
                print("⚠️  未找到图像文件，将使用模拟数据进行测试")
                val_images = None
                val_labels = None
            else:
                print(f"✓ 总共加载 {len(val_images)} 张验证图像")
        elif data_dir and not os.path.exists(data_dir):
            print(f"⚠️  数据目录不存在: {data_dir}，将使用模拟数据进行测试")
            val_images = None
            val_labels = None
    
    if load_mode == '3' or (not val_images and not val_labels):
        print("⚠️  跳过图像测试，将使用模拟数据进行部分测试")
        val_images = None
        val_labels = None
    
    # ========== 选择初始 Prompts 来源 ==========
    print("\n初始 Prompts 来源:")
    print("  1. 从文件加载（推荐，使用固定的 prompts）")
    print("  2. 使用 LLM 生成（每次运行都会生成新的 prompts）")
    
    prompts_mode = input("请选择 Prompts 来源 (1/2, 默认1): ").strip()
    if not prompts_mode:
        prompts_mode = '1'
    
    initial_prompts_file = None
    if prompts_mode == '1':
        prompts_file = input("请输入初始 Prompts JSON 文件路径（留空使用默认: test_initial_prompts.json）: ").strip()
        if not prompts_file:
            prompts_file = 'test_initial_prompts.json'
        
        if os.path.exists(prompts_file):
            initial_prompts_file = prompts_file
            print(f"✓ 将使用文件中的 Prompts: {prompts_file}")
        else:
            print(f"⚠️  文件不存在: {prompts_file}，将使用 LLM 生成 Prompts")
            initial_prompts_file = None
    else:
        print("将使用 LLM 生成初始 Prompts")
    
    # ========== Step 2: 测试模型加载 ==========
    print("\n【Step 2】测试模型加载")
    print("-" * 80)
    
    test_model = input("是否测试模型加载？(y/n, 默认y): ").strip().lower()
    if test_model != 'n':
        try:
            from models.clip import CLIPModel
            
            print(f"尝试加载模型: {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 读取配置
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    image_encoder = config.get('image_encoder', 'resnet18')
                    text_encoder = config.get('text_encoder', 'clip:ViT-B/32')
                    embed_dim = config.get('embed_dim', 512)
                    temperature = config.get('temperature', 0.07)
                else:
                    image_encoder = 'resnet18'
                    text_encoder = 'clip:ViT-B/32'
                    embed_dim = 512
                    temperature = 0.07
                
                print(f"  图像编码器: {image_encoder}")
                print(f"  文本编码器: {text_encoder}")
                print(f"  嵌入维度: {embed_dim}")
                print(f"  温度参数: {temperature}")
                
                model = CLIPModel(
                    image_encoder_name=image_encoder,
                    text_encoder_name=text_encoder,
                    embed_dim=embed_dim,
                    temperature=temperature
                )
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                print("✓ 模型加载成功")
            else:
                print(f"⚠️  Checkpoint 文件不存在: {checkpoint_path}")
                print("  将创建新模型进行测试")
                model = CLIPModel(
                    image_encoder_name='resnet18',
                    text_encoder_name='clip:ViT-B/32',
                    embed_dim=512,
                    temperature=0.07
                )
                print("✓ 新模型创建成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ========== Step 3: 测试 LLM API ==========
    print("\n【Step 3】测试 LLM API")
    print("-" * 80)
    print(f"使用API: 硅基流动 (SiliconFlow)")
    print(f"模型: {DEEPSEEK_MODEL}")
    print(f"Base URL: {DEEPSEEK_BASE_URL}")
    
    test_llm = input("是否测试 LLM API？(y/n, 默认y): ").strip().lower()
    if test_llm != 'n':
        try:
            if OPENAI_AVAILABLE:
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL
                )
                print("✓ DeepSeek API 客户端初始化成功")
                
                # 测试调用
                test_response = input("是否发送测试请求？(y/n, 默认y): ").strip().lower()
                if test_response != 'n':
                    print("发送测试请求...")
                    try:
                        response = client.chat.completions.create(
                            model=DEEPSEEK_MODEL,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "请用一句话描述什么是X光片。"}
                            ],
                            temperature=0.7,
                            max_tokens=1024,
                            stream=False
                        )
                        result = response.choices[0].message.content
                        print(f"✓ LLM 响应: {result}")
                    except Exception as api_error:
                        error_msg = str(api_error)
                        # 提供更友好的错误信息
                        if "402" in error_msg or "Insufficient Balance" in error_msg or "余额" in error_msg:
                            print(f"❌ API调用失败: 账户余额不足")
                            print("   请检查硅基流动账户余额: https://siliconflow.cn")
                        elif "401" in error_msg or "Unauthorized" in error_msg or "认证" in error_msg:
                            print(f"❌ API调用失败: API密钥无效或已过期")
                            print("   请检查API密钥是否正确")
                        elif "429" in error_msg or "Rate limit" in error_msg or "限流" in error_msg:
                            print(f"❌ API调用失败: 请求频率过高，请稍后重试")
                        else:
                            print(f"❌ API调用失败: {error_msg}")
                        import traceback
                        traceback.print_exc()
            else:
                print("⚠️  OpenAI SDK 未安装，无法测试 LLM API")
                print("   请安装: pip install openai")
        except Exception as e:
            print(f"❌ LLM API 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== Step 4: 测试图像加载和预处理 ==========
    print("\n【Step 4】测试图像加载和预处理")
    print("-" * 80)
    
    test_image = input("是否测试图像加载？(y/n, 默认y): ").strip().lower()
    if test_image != 'n' and val_images and len(val_images) > 0:
        try:
            from PIL import Image
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            test_img_path = val_images[0]
            print(f"测试图像: {test_img_path}")
            
            image = Image.open(test_img_path).convert('RGB')
            print(f"  原始图像尺寸: {image.size}")
            
            image_tensor = transform(image)
            print(f"  预处理后张量形状: {image_tensor.shape}")
            print(f"  张量范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            
            print("✓ 图像加载和预处理测试成功")
        except Exception as e:
            print(f"❌ 图像加载测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  跳过图像加载测试（无图像数据）")
    
    # ========== Step 5: 测试初始化 Prompts（不创建完整优化器）==========
    print("\n【Step 5】测试 LLM 生成初始 Prompts")
    print("-" * 80)
    
    test_init = input("是否测试生成初始 Prompts？(y/n, 默认y): ").strip().lower()
    if test_init != 'n':
        try:
            # 创建模拟的 LLM API 函数
            def mock_llm_api(system=None, user=None):
                if not OPENAI_AVAILABLE:
                    return f"Mock description for: {user[:50] if user else 'test'}"
                
                try:
                    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                    response = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[
                            {"role": "system", "content": system or "You are a medical expert."},
                            {"role": "user", "content": user or "Test"}
                        ],
                        temperature=0.7,
                        max_tokens=1024,
                        stream=False
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    error_msg = str(e)
                    if "402" in error_msg or "Insufficient Balance" in error_msg or "余额" in error_msg:
                        print(f"  LLM API 调用失败: 账户余额不足")
                    elif "401" in error_msg or "Unauthorized" in error_msg:
                        print(f"  LLM API 调用失败: API密钥无效")
                    else:
                        print(f"  LLM API 调用失败: {error_msg[:100]}")
                    return f"Mock description (API error: {str(e)[:50]})"
            
            # 测试生成一个类别的 Prompt
            test_class = class_names[0]
            print(f"测试生成类别 '{test_class}' 的 Prompt...")
            
            prompt = mock_llm_api(
                system="你是骨科影像专家。请用英文生成详细、准确的医学影像描述。",
                user=f"请生成一段用于 CLIP 模型的详细英文描述，描述髋关节术后 X 光片中 '{test_class}' 类别的视觉特征。"
                     f"要求：1) 准确描述该类别的医学特征；2) 控制在 50 个词以内；3) 使用专业但易懂的术语。"
            )
            
            print(f"✓ 生成的 Prompt:")
            print(f"  {prompt}")
            
            # 询问是否生成所有类别
            generate_all = input(f"\n是否生成所有 {len(class_names)} 个类别的 Prompts？(y/n, 默认n): ").strip().lower()
            if generate_all == 'y':
                prompts = {}
                for cls in tqdm(class_names, desc="生成 Prompts"):
                    prompt = mock_llm_api(
                        system="你是骨科影像专家。请用英文生成详细、准确的医学影像描述。",
                        user=f"请生成一段用于 CLIP 模型的详细英文描述，描述髋关节术后 X 光片中 '{cls}' 类别的视觉特征。"
                             f"要求：1) 准确描述该类别的医学特征；2) 控制在 50 个词以内；3) 使用专业但易懂的术语。"
                    )
                    prompts[cls] = prompt.strip()
                    print(f"  {cls}: {prompts[cls][:80]}...")
                
                # 保存生成的 Prompts
                save_prompts = input("\n是否保存生成的 Prompts？(y/n, 默认y): ").strip().lower()
                if save_prompts != 'n':
                    output_file = 'test_initial_prompts.json'
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'class_names': class_names,
                            'prompts': prompts
                        }, f, indent=2, ensure_ascii=False)
                    print(f"✓ Prompts 已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 生成 Prompts 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== Step 6: 测试评估功能（需要模型和图像）==========
    print("\n【Step 6】测试评估功能")
    print("-" * 80)
    
    test_eval = input("是否测试评估功能？(y/n, 默认n): ").strip().lower()
    if test_eval == 'y' and val_images and len(val_images) > 0 and os.path.exists(checkpoint_path):
        try:
            print("创建优化器实例（仅用于测试评估功能）...")
            
            # 创建模拟的 LLM API（不实际调用，使用预设的 Prompts）
            def simple_llm_api(system=None, user=None):
                # 返回简单的预设描述
                if 'Acetabular Loosening' in user:
                    return "The acetabular cup shows signs of loosening with radiolucent lines."
                elif 'Dislocation' in user:
                    return "The prosthetic hip joint has lost congruency with dislocation."
                elif 'Fracture' in user:
                    return "There is discontinuity of cortical bone with periprosthetic fracture."
                else:
                    return "A medical condition visible on X-ray imaging."
            
            # 创建优化器（但跳过初始化 Prompts）
            optimizer = AutoPromptOptimizer.__new__(AutoPromptOptimizer)
            optimizer.class_names = class_names
            optimizer.val_images = val_images
            optimizer.val_labels = np.array(val_labels)
            optimizer.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            optimizer.img_size = 224
            optimizer.llm_api_func = simple_llm_api
            
            # 图像预处理
            optimizer.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 加载模型（使用之前测试过的模型或重新加载）
            from models.clip import CLIPModel
            checkpoint = torch.load(checkpoint_path, map_location=optimizer.device)
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                image_encoder = config.get('image_encoder', 'resnet18')
                text_encoder = config.get('text_encoder', 'clip:ViT-B/32')
                embed_dim = config.get('embed_dim', 512)
                temperature = config.get('temperature', 0.07)
            else:
                image_encoder = 'resnet18'
                text_encoder = 'clip:ViT-B/32'
                embed_dim = 512
                temperature = 0.07
            
            optimizer.model = CLIPModel(
                image_encoder_name=image_encoder,
                text_encoder_name=text_encoder,
                embed_dim=embed_dim,
                temperature=temperature
            )
            
            if 'model_state_dict' in checkpoint:
                optimizer.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                optimizer.model.load_state_dict(checkpoint)
            
            optimizer.model.eval()
            optimizer.model.to(optimizer.device)
            
            # 创建测试 Prompts
            test_prompts = {
                cls: f"Medical description for {cls} visible on X-ray imaging."
                for cls in class_names
            }
            
            print(f"使用 {len(val_images)} 张图像进行测试评估...")
            acc, cm = optimizer.evaluate(test_prompts)
            
            print(f"\n✓ 评估完成")
            print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  混淆矩阵形状: {cm.shape}")
            print(f"  混淆矩阵:\n{cm}")
            
        except Exception as e:
            print(f"❌ 评估功能测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  跳过评估功能测试（需要图像数据和模型）")
    
    # ========== Step 7: 测试混淆矩阵分析 ==========
    print("\n【Step 7】测试混淆矩阵分析")
    print("-" * 80)
    
    test_confusion = input("是否测试混淆矩阵分析？(y/n, 默认y): ").strip().lower()
    if test_confusion != 'n':
        try:
            # 创建模拟混淆矩阵
            num_classes = len(class_names)
            mock_cm = np.random.randint(0, 10, size=(num_classes, num_classes))
            # 确保对角线有较大的值（正确预测）
            np.fill_diagonal(mock_cm, np.random.randint(20, 50, size=num_classes))
            
            print("模拟混淆矩阵:")
            print(mock_cm)
            
            # 测试 find_worst_confusion 函数
            optimizer_test = AutoPromptOptimizer.__new__(AutoPromptOptimizer)
            optimizer_test.class_names = class_names
            
            actual_class, predicted_class, error_count = optimizer_test.find_worst_confusion(mock_cm)
            
            print(f"\n✓ 混淆分析完成")
            print(f"  最严重混淆: {actual_class} -> {predicted_class}")
            print(f"  错误次数: {error_count}")
            
            # 测试打印混淆摘要
            optimizer_test._print_confusion_summary(mock_cm)
            
        except Exception as e:
            print(f"❌ 混淆矩阵分析测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== Step 8: 完整流程测试（可选）==========
    print("\n【Step 8】完整流程测试")
    print("-" * 80)
    
    test_full = input("是否运行完整的优化流程测试？（需要模型、图像和 LLM API）(y/n, 默认n): ").strip().lower()
    if test_full == 'y' and val_images and len(val_images) > 0 and os.path.exists(checkpoint_path):
        try:
            print("创建完整的优化器实例...")
            
            optimizer = AutoPromptOptimizer(
                class_names=class_names,
                val_images=val_images,
                val_labels=val_labels,
                checkpoint_path=checkpoint_path,
                image_encoder_name='resnet18',
                text_encoder_name='clip:ViT-B/32',
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                initial_prompts_file=initial_prompts_file
            )
            
            # 运行优化（仅测试 1-2 轮）
            max_rounds = input("输入最大优化轮数（默认2）: ").strip()
            max_rounds = int(max_rounds) if max_rounds.isdigit() else 2
            
            print(f"\n开始运行优化流程（将持续运行直到所有混淆对都尝试过，耐心值: 10）...")
            best_prompts = optimizer.optimize_loop(
                max_rounds=max_rounds,  # 参数保留但不使用
                min_improvement=0.001,
                patience=10
            )
            
            # 保存结果
            save_file = 'test_optimized_prompts.json'
            optimizer.save_prompts(save_file)
            print(f"\n✓ 完整流程测试完成，结果已保存到: {save_file}")
            
        except Exception as e:
            print(f"❌ 完整流程测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  跳过完整流程测试")
    
    print("\n" + "="*80)
    print("分步测试完成！")
    print("="*80)


# ==================== 直接运行完整优化流程 ====================

def run_optimization(
    checkpoint_path=None,
    train_output_dir=None,
    fold_num=None,
    data_dir=None,
    initial_prompts_file='test_initial_prompts.json',
    max_rounds=10,
    min_improvement=0.001,
    patience=10,
    output_file='optimized_prompts.json',
    device='cuda:0'
):
    """
    直接运行完整的优化流程，跳过所有测试步骤
    
    Args:
        checkpoint_path: 模型checkpoint路径（如果为None，会尝试自动检测）
        train_output_dir: 训练输出目录（如果为None，会从checkpoint路径推断）
        fold_num: fold编号（如果为None，会从checkpoint路径推断）
        data_dir: 数据目录（如果为None，使用默认路径）
        initial_prompts_file: 初始prompts文件路径
        max_rounds: 最大优化轮数
        min_improvement: 最小改进阈值
        patience: 早停耐心值
        output_file: 输出文件路径
        device: 设备
    """
    import glob
    
    print("="*80)
    print("AutoPromptOptimizer - 直接运行完整优化流程")
    print("="*80)
    
    # 1. 设置默认checkpoint路径
    if checkpoint_path is None:
        default_checkpoint = 'checkpoints/clip_models/resnet18_clip_ViT-B_32/fold_4/checkpoint_best.pth'
        if os.path.exists(default_checkpoint):
            checkpoint_path = default_checkpoint
            print(f"✓ 使用默认checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = input("请输入模型 checkpoint 路径: ").strip()
            if not checkpoint_path:
                print("❌ 未提供checkpoint路径，退出")
                return
    
    # 2. 从checkpoint读取类别信息
    class_names = None
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
                print(f"✓ 从checkpoint读取类别信息: {len(class_names)} 个类别")
        except Exception as e:
            print(f"⚠️  读取checkpoint失败: {e}")
    
    if class_names is None:
        class_names = [
            'Acetabular Loosening', 'Dislocation', 'Fracture', 'Good Place', 
            'Infection', 'Native Hip', 'Spacer', 'Stem Loosening', 'Wear'
        ]
        print(f"使用默认类别名称: {len(class_names)} 个类别")
    
    # 3. 加载验证集
    print("\n加载验证集...")
    val_images = []
    val_labels = []
    
    # 设置默认数据目录
    if data_dir is None:
        data_dir = '/home/ln/wangweicheng/ModelsTotrain/single_label_data'
    
    # 尝试从fold加载
    if train_output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if 'fold_' in checkpoint_dir:
            train_output_dir = os.path.dirname(checkpoint_dir)
            if fold_num is None:
                fold_name = os.path.basename(checkpoint_dir)
                if fold_name.startswith('fold_'):
                    try:
                        fold_num = int(fold_name.split('fold_')[-1])
                    except:
                        pass
    
    if train_output_dir and fold_num and data_dir:
        try:
            val_images, val_labels, class_names = load_validation_set_from_fold(
                train_output_dir, fold_num, data_dir, random_state=42
            )
            print(f"✓ 从Fold {fold_num}加载验证集: {len(val_images)} 张图像")
        except Exception as e:
            print(f"⚠️  从fold加载失败: {e}，尝试从数据目录加载...")
            train_output_dir = None
    
    # 如果从fold加载失败，尝试从数据目录加载
    if not val_images:
        if os.path.exists(data_dir):
            try:
                val_images, val_labels, class_names = load_validation_set_from_dir(data_dir, class_names)
                print(f"✓ 从数据目录加载验证集: {len(val_images)} 张图像")
            except Exception as e:
                print(f"❌ 从数据目录加载失败: {e}")
                return
        else:
            print(f"❌ 数据目录不存在: {data_dir}")
            return
    
    if len(val_images) == 0:
        print("❌ 未找到验证图像，退出")
        return
    
    # 4. 检查初始prompts文件
    if not os.path.exists(initial_prompts_file):
        print(f"⚠️  初始prompts文件不存在: {initial_prompts_file}")
        use_file = input("是否使用LLM生成初始prompts？(y/n, 默认n): ").strip().lower()
        if use_file != 'y':
            print("退出")
            return
        initial_prompts_file = None
    else:
        print(f"✓ 将使用初始prompts文件: {initial_prompts_file}")
    
    # 5. 创建优化器
    print("\n创建优化器...")
    try:
        optimizer = AutoPromptOptimizer(
            class_names=class_names,
            val_images=val_images,
            val_labels=val_labels,
            checkpoint_path=checkpoint_path,
            device=device,
            initial_prompts_file=initial_prompts_file
        )
        print("✓ 优化器创建成功")
    except Exception as e:
        print(f"❌ 创建优化器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 运行优化
    print(f"\n开始优化流程（将持续运行直到所有混淆对都尝试过，耐心值: {patience}）...")
    try:
        best_prompts = optimizer.optimize_loop(
            max_rounds=max_rounds,  # 参数保留但不使用
            min_improvement=min_improvement,
            patience=patience
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断优化流程")
        best_prompts = optimizer.best_prompts or optimizer.current_prompts
    except Exception as e:
        print(f"❌ 优化流程失败: {e}")
        import traceback
        traceback.print_exc()
        best_prompts = optimizer.best_prompts or optimizer.current_prompts
    
    # 7. 保存结果
    if best_prompts:
        optimizer.save_prompts(output_file)
        print(f"\n✓ 优化完成，结果已保存到: {output_file}")
        print(f"  最终准确率: {optimizer.best_score:.4f} ({optimizer.best_score*100:.2f}%)")
    else:
        print("\n⚠️  未获得优化结果")


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    
    # 1. 定义 LLM API 调用函数（需要用户自己实现）
    def my_llm_api(system=None, user=None):
        # 这里集成实际的 LLM API，例如 OpenAI、本地模型等
        # import openai
        # response = openai.ChatCompletion.create(...)
        # return response.choices[0].message.content
        pass
    
    # 2. 准备数据
    class_names = ['Normal', 'Loosening', 'Dislocation', 'Fracture', 'Infection', 
                   'Osteolysis', 'Heterotopic', 'Periprosthetic', 'Other']
    
    # 验证集图像路径列表
    val_images = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
    
    # 验证集标签（整数索引，对应 class_names 的顺序）
    val_labels = [0, 1, 2, ...]
    
    # 3. 创建优化器
    optimizer = AutoPromptOptimizer(
        class_names=class_names,
        val_images=val_images,
        val_labels=val_labels,
        checkpoint_path='checkpoints/clip_models/resnet18_clip_ViT-B_32/fold_4/checkpoint_best.pth',
        image_encoder_name='resnet18',
        text_encoder_name='clip:ViT-B/32',
        device='cuda:0',
        llm_api_func=my_llm_api
    )
    
    # 4. 运行优化
    best_prompts = optimizer.optimize_loop(max_rounds=10, min_improvement=0.001, patience=3)
    
    # 5. 保存结果
    optimizer.save_prompts('optimized_prompts.json')
    
    # 6. 使用优化后的 Prompts 进行最终评估
    final_acc, final_cm = optimizer.evaluate(best_prompts)
    print(f"\n最终验证准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoPromptOptimizer - 自动优化CLIP模型的文本描述')
    parser.add_argument('--test', action='store_true', help='运行分步测试模式')
    parser.add_argument('--run', action='store_true', help='直接运行完整优化流程（跳过所有测试）')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型checkpoint路径')
    parser.add_argument('--train-output-dir', type=str, default=None, help='训练输出目录（包含fold_N子目录）')
    parser.add_argument('--fold', type=int, default=None, help='fold编号（1-5）')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录（按类别组织的文件夹）')
    parser.add_argument('--initial-prompts', type=str, default='test_initial_prompts.json', help='初始prompts文件路径')
    parser.add_argument('--max-rounds', type=int, default=10, help='最大优化轮数（默认10）')
    parser.add_argument('--min-improvement', type=float, default=0.001, help='最小改进阈值（默认0.001）')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值（默认10）')
    parser.add_argument('--output', type=str, default='optimized_prompts.json', help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备（默认cuda:0）')
    
    args = parser.parse_args()
    
    if args.run:
        # 直接运行完整优化流程
        run_optimization(
            checkpoint_path=args.checkpoint,
            train_output_dir=args.train_output_dir,
            fold_num=args.fold,
            data_dir=args.data_dir or '/home/ln/wangweicheng/ModelsTotrain/single_label_data',
            initial_prompts_file=args.initial_prompts,
            max_rounds=args.max_rounds,
            min_improvement=args.min_improvement,
            patience=args.patience,
            output_file=args.output,
            device=args.device
        )
    elif args.test:
        # 运行分步测试
        step_by_step_test()
    else:
        # 默认运行完整优化流程（交互式）
        print("="*80)
        print("AutoPromptOptimizer")
        print("="*80)
        print("\n使用方式:")
        print("  1. 直接运行完整优化流程: python auto_prompt_optimizer.py --run")
        print("  2. 运行分步测试: python auto_prompt_optimizer.py --test")
        print("  3. 交互式运行完整流程: python auto_prompt_optimizer.py")
        print()
        
        choice = input("请选择运行模式 (1=直接运行/2=分步测试/3=交互式运行, 默认1): ").strip()
        if not choice:
            choice = '1'
        
        if choice == '1':
            # 直接运行完整优化流程
            run_optimization()
        elif choice == '2':
            # 运行分步测试
            step_by_step_test()
        else:
            # 交互式运行完整流程
            run_optimization()

