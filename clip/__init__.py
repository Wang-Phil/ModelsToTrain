# CLIP package for PMC-CLIP support
# 这个包主要用于提供 pmcclip 模块
# 标准 CLIP 库应该从系统安装的 clip 包导入

# 不在这里导入标准 CLIP，让其他模块直接从系统安装的 clip 包导入
# 只导出 pmcclip 子模块
__all__ = []
