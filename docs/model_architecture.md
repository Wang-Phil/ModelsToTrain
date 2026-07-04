# SupCon-CLIP 模型架构图

## 流程图 (Mermaid)

可在支持 Mermaid 的编辑器中渲染（如 VS Code 插件、Typora、GitHub 等），或使用 https://mermaid.live 导出为 PNG/SVG。

```mermaid
flowchart TB
    subgraph Input["输入"]
        I["图像 X (B×3×224×224)"]
        T["配对文本 (batch)"]
        C["类别文本 (K 类)"]
    end

    subgraph Encoders["编码器"]
        IE["图像编码器\n(ResNet18)"]
        TE["文本编码器\n(CLIP ViT-B/32)"]
    end

    subgraph Features["特征"]
        Fimg["图像特征 z_i\n(B×512)"]
        Ftxt["配对文本特征 z_t\n(B×512)"]
        Fcls["类别文本特征 z_c\n(K×512)"]
    end

    subgraph Losses["损失函数"]
        Lsup["SupCon Loss\n(单视图, 同类拉近)"]
        Lclip["CLIP Loss\n(图像-文本对齐)"]
        Lcls["Classification Loss\n(Focal / CE)"]
    end

    I --> IE
    IE --> Fimg
    T --> TE
    TE --> Ftxt
    C --> TE
    TE --> Fcls

    Fimg --> Lsup
    Fimg --> Lclip
    Ftxt --> Lclip
    Fimg --> Lcls
    Fcls --> Lcls

    Lsup --> Total["L = λ₁·L_supcon + λ₂·L_clip + λ₃·L_cls"]
    Lclip --> Total
    Lcls --> Total
```

## 简化框图 (左右双分支)

```mermaid
flowchart LR
    subgraph Image["图像分支"]
        A[图像] --> B[ResNet18]
        B --> C["z_i (512-d)"]
    end

    subgraph Text["文本分支"]
        D[配对文本] --> E[CLIP ViT-B/32]
        E --> F["z_t (512-d)"]
        G[类别文本] --> E
        E --> H["z_c (K×512)"]
    end

    C --> I[SupCon]
    C --> J[CLIP]
    F --> J
    C --> K[分类器]
    H --> K
    I --> L[总损失]
    J --> L
    K --> L
```

## 模块说明

| 模块 | 说明 |
|------|------|
| 图像编码器 | ResNet18，输出 512 维 L2 归一化特征 |
| 文本编码器 | CLIP ViT-B/32 文本塔，输出 512 维 |
| SupCon | 单视图有监督对比：正样本为 batch 内同类别样本 |
| CLIP Loss | 对称 CE(image→text, text→image)，温度 τ=0.07 |
| 分类 | 图像特征与 K 类文本特征相似度 → logits → Focal Loss |
