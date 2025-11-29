# Git 常用操作指南

## 1. 仓库初始化和克隆

```bash
# 初始化新仓库
git init

# 克隆远程仓库
git clone <仓库URL>
git clone https://github.com/username/repo.git

# 克隆到指定目录
git clone <仓库URL> <目录名>
```

## 2. 基本操作（查看状态、添加、提交）

```bash
# 查看工作区状态
git status

# 查看简洁状态
git status -s

# 查看文件差异
git diff                    # 工作区与暂存区的差异
git diff --staged           # 暂存区与最新提交的差异
git diff HEAD               # 工作区与最新提交的差异

# 添加文件到暂存区
git add <文件名>            # 添加单个文件
git add .                   # 添加所有修改的文件
git add *.py                # 添加匹配模式的文件
git add -A                  # 添加所有变更（包括删除）

# 提交
git commit -m "提交信息"
git commit -am "提交信息"   # 跳过暂存区，直接提交已跟踪的文件

# 修改最近一次提交
git commit --amend          # 修改提交信息或添加遗漏的文件
git commit --amend --no-edit # 只添加文件，不修改提交信息
```

## 3. 查看历史记录

```bash
# 查看提交历史
git log                     # 详细日志
git log --oneline           # 单行显示
git log --graph             # 图形化显示分支
git log --all --graph --oneline  # 所有分支的图形化单行显示
git log -n 5                # 只显示最近5次提交

# 查看某个文件的修改历史
git log <文件名>
git log -p <文件名>         # 显示每次修改的详细内容

# 查看谁修改了文件的哪一行
git blame <文件名>
```

## 4. 撤销操作

```bash
# 撤销工作区的修改（危险操作，未提交的修改会丢失）
git checkout -- <文件名>
git restore <文件名>        # Git 2.23+ 新命令

# 撤销暂存区的文件（取消add）
git reset HEAD <文件名>
git restore --staged <文件名>  # Git 2.23+ 新命令

# 撤销最近一次提交（保留修改）
git reset --soft HEAD~1     # 撤销提交，保留暂存区
git reset --mixed HEAD~1    # 撤销提交和暂存，保留工作区（默认）
git reset --hard HEAD~1     # 危险！撤销提交、暂存和工作区

# 恢复到指定提交
git reset --hard <commit_hash>

# 恢复单个文件到指定版本
git checkout <commit_hash> -- <文件名>
```

## 5. 分支操作

```bash
# 查看分支
git branch                  # 本地分支
git branch -a               # 所有分支（包括远程）
git branch -r               # 远程分支

# 创建分支
git branch <分支名>
git checkout -b <分支名>    # 创建并切换到新分支
git switch -c <分支名>      # Git 2.23+ 新命令

# 切换分支
git checkout <分支名>
git switch <分支名>         # Git 2.23+ 新命令

# 删除分支
git branch -d <分支名>      # 删除已合并的分支
git branch -D <分支名>      # 强制删除分支

# 合并分支
git merge <分支名>          # 将指定分支合并到当前分支
git merge --no-ff <分支名>  # 不使用fast-forward合并

# 变基（rebase）
git rebase <分支名>         # 将当前分支变基到指定分支
git rebase -i HEAD~3        # 交互式变基，修改最近3次提交
```

## 6. 远程仓库操作

```bash
# 查看远程仓库
git remote                  # 列出远程仓库
git remote -v               # 显示远程仓库URL

# 添加远程仓库
git remote add <名称> <URL>
git remote add origin https://github.com/username/repo.git

# 删除远程仓库
git remote remove <名称>

# 修改远程仓库URL
git remote set-url <名称> <新URL>

# 获取远程更新（不合并）
git fetch <远程名称>
git fetch origin

# 拉取并合并
git pull                    # 拉取并合并当前分支
git pull origin <分支名>
git pull --rebase           # 使用rebase方式拉取

# 推送
git push                    # 推送当前分支
git push origin <分支名>
git push -u origin <分支名> # 设置上游分支并推送
git push --all              # 推送所有分支
git push --tags             # 推送所有标签

# 删除远程分支
git push origin --delete <分支名>
```

## 7. 标签操作

```bash
# 查看标签
git tag                     # 列出所有标签
git tag -l "v1.*"           # 列出匹配模式的标签

# 创建标签
git tag <标签名>            # 轻量标签
git tag -a <标签名> -m "标签说明"  # 附注标签
git tag <标签名> <commit_hash>  # 给指定提交打标签

# 推送标签
git push origin <标签名>    # 推送单个标签
git push --tags             # 推送所有标签

# 删除标签
git tag -d <标签名>         # 删除本地标签
git push origin --delete <标签名>  # 删除远程标签
```

## 8. 暂存操作（Stash）

```bash
# 暂存当前修改
git stash                   # 暂存修改
git stash save "说明信息"   # 带说明的暂存
git stash -u                # 包括未跟踪的文件

# 查看暂存列表
git stash list

# 恢复暂存
git stash apply             # 恢复但不删除stash
git stash pop               # 恢复并删除stash
git stash apply stash@{0}   # 恢复指定的stash

# 删除暂存
git stash drop              # 删除最近一个
git stash drop stash@{0}    # 删除指定的
git stash clear             # 清空所有stash

# 查看暂存内容
git stash show
git stash show -p           # 显示详细差异
```

## 9. 比较和查找

```bash
# 比较两个分支
git diff <分支1> <分支2>
git diff master..develop    # 比较两个分支

# 查找包含特定内容的提交
git log -S "关键词"         # 查找修改了特定内容的提交
git log --grep="关键词"     # 在提交信息中搜索

# 查找文件
git ls-files                # 列出所有跟踪的文件
git ls-files | grep "关键词"
```

## 10. 配置操作

```bash
# 查看配置
git config --list           # 所有配置
git config user.name        # 查看用户名
git config user.email       # 查看邮箱

# 设置配置（全局）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 设置配置（当前仓库）
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 设置默认编辑器
git config --global core.editor vim

# 设置默认分支名
git config --global init.defaultBranch main

# 设置别名（快捷命令）
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
```

## 11. 实用技巧

```bash
# 查看文件大小
git ls-files | xargs du -h | sort -h

# 清理未跟踪的文件
git clean -n                # 预览将要删除的文件
git clean -f                # 删除未跟踪的文件
git clean -fd               # 删除未跟踪的文件和目录

# 忽略文件（.gitignore）
echo "*.log" >> .gitignore
echo "*.pyc" >> .gitignore
echo "node_modules/" >> .gitignore

# 查看仓库大小
git count-objects -vH

# 压缩仓库
git gc --aggressive --prune=now
```

## 12. 常用工作流程示例

### 日常开发流程
```bash
# 1. 更新本地代码
git pull origin main

# 2. 创建功能分支
git checkout -b feature/new-feature

# 3. 开发并提交
git add .
git commit -m "Add new feature"

# 4. 推送分支
git push -u origin feature/new-feature

# 5. 合并到主分支（在GitHub/GitLab上创建PR/MR，或本地合并）
git checkout main
git merge feature/new-feature
git push origin main

# 6. 删除功能分支
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

### 紧急修复Bug
```bash
# 1. 从主分支创建修复分支
git checkout -b hotfix/bug-fix

# 2. 修复并提交
git add .
git commit -m "Fix critical bug"

# 3. 合并到主分支和开发分支
git checkout main
git merge hotfix/bug-fix
git push origin main

git checkout develop
git merge hotfix/bug-fix
git push origin develop
```

## 13. 常见问题处理

```bash
# 解决合并冲突
git status                  # 查看冲突文件
# 手动编辑冲突文件，解决冲突后：
git add <冲突文件>
git commit

# 撤销已推送的提交（危险，谨慎使用）
git revert <commit_hash>    # 创建新提交来撤销指定提交（推荐）
git reset --hard <commit_hash>
git push --force            # 强制推送（不推荐，会改写历史）

# 找回误删的文件
git log --all --full-history -- <文件路径>
git checkout <commit_hash>^ -- <文件路径>

# 修改提交信息
git rebase -i HEAD~3        # 交互式修改最近3次提交信息
# 在弹出的编辑器中，将 pick 改为 reword，保存后修改提交信息
```

## 14. 推荐配置

```bash
# 设置常用的全局配置
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
git config --global core.editor vim
git config --global color.ui true

# 设置常用别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
```

## 15. .gitignore 常用模板

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.log

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt
checkpoints/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# 数据文件（大文件）
*.h5
*.hdf5
data/
datasets/
```

