# GitHub SSH 配置（jlu_final_result）

## 公钥内容（完整一行）

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ3MRUkV4nECWU+A0l4Sam3rVAexXL8XKVCY70EMrxgC github-jlu_final_result
```

复制上面整行到 GitHub。

## 配置步骤

1. 打开 [GitHub SSH keys 设置](https://github.com/settings/keys)
2. 点击 **New SSH key**
3. Title 可填：`jlu_final_result` 或本机名称
4. Key type：**Authentication Key**
5. 将上方公钥整行粘贴到 Key 字段
6. 点击 **Add SSH key**

## 验证命令

```bash
ssh -T git@github.com
```

成功时会看到类似：`Hi <username>! You've successfully authenticated...`

## 推送命令

```bash
cd /home/ln/wangweicheng/ModelsTotrain
git push -u jlu_final jlu_final_release:main
git lfs push jlu_final jlu_final_release --all
```

## 密钥文件路径

- 私钥：`~/.ssh/id_ed25519_github`（勿泄露、勿提交到仓库）
- 公钥：`~/.ssh/id_ed25519_github.pub`

`~/.ssh/config` 已为 `github.com` 指定该私钥（`IdentitiesOnly yes`），不影响现有 `aliyun` 配置。
