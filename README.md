项目概览
-
本仓库包含多个医学图像/模型训练相关的脚本、配置与数据目录。此 README 用于说明各个顶层文件/目录的用途及协作规则，便于团队协作并避免将大数据直接推送到远程仓库。

目录与规则概览
-
- `*.py`, `trainers/`, `scripts/`：可执行代码与训练脚本，按需修改并提交代码变更。
- `configs/`, `open_clip/`, `ModelsTotrain/`：配置与源码目录，提交前请保持配置清晰并添加变更说明。
- `data/`, `old_data/`, `new_data/`, `eltra_test/`：数据目录。**注意：这些目录的内容默认被 `.gitignore` 忽略，仓库仅保留 `.gitkeep` 占位文件以保存目录结构**。不要直接将原始数据或大文件提交到主仓库。
- `.gitattributes`：如果需要上传数据，请使用 Git LFS（管理员配置后分批次上传）。大型数据上传应先在私有存储或 release/数据仓库中管理，再在仓库中引用元信息。

提交与分支策略
-
- 开发请使用 feature 分支：`git checkout -b feat/xxx` 或 `git checkout -b fix/xxx`。
- 只提交代码和小的文本/配置文件到主仓库。数据文件不要提交到主分支。
- 需要提交目录结构时，使用 `.gitkeep` 占位符并在 `.gitignore` 中保持数据文件被忽略。
- 轻量变更（仅 README、配置、占位符）可直接推送到远端分支并发起 PR；大文件或历史重写需通过管理员协调。

数据上传建议（可选，管理员操作）
-
- 若确需把数据推到仓库，请先启用 Git LFS 并注册 LFS 路径到 `.gitattributes`。
- 推荐通过 SSH+公钥方式推送大文件，或将数据拆分成非常小的批次并单独推送。
- 如果 HTTPS 推送在写大 pack 时失败（HTTP 500），请改用 SSH 或联系 GitHub 支持/管理员。

协作与联系方式
-
- 提交 PR 前请在 PR 描述中写明变更内容与影响范围（数据、模型、训练命令等）。
- 如需上传数据或更改 LFS 策略，请联系仓库管理员（在 README 或团队频道标注具体联系人）。

附：快速操作命令
-
- 创建分支并提交：
- `git checkout -b feat/your-change`
- `git add <files>`
- `git commit -m "Describe change"`
- `git push origin feat/your-change`

谢谢。
