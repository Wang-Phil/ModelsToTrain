# cv_summary.json 全量索引

自动生成：共 **171** 个文件（相对仓库根目录路径）。

重新生成：在仓库根目录执行 `python3 docs/generate_cv_summary_index.py`。

分类规则（按顺序匹配第一条）：`output/` → `new_data_models` → `ablation_study` → `clip_models` / `clip_agent_v2` / `comparison_experiments` → `final_models` / `final_starnet_models` → 其余归入 **other**。

同级 CSV：`docs/all_cv_summary_index.csv`（列：`category,relative_path`）。

## output（20）

- `output/biomedcoop_pmcclip/cv_summary.json`
- `output/hybrid_coop_sccm/cv_summary.json`
- `output/hybrid_coop_sccm_new_loss/cv_summary.json`
- `output/hybrid_original_clip_resnet50_biomedclip/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_ce_loss_only/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_contrastive_loss_only/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_coop/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_with_CLASSIFICATION_LOSS_WEIGHT=1/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_with_CONTRASTIVE_LOSS_WEIGHT=1/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_with_distillation/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_with_distillation_new/cv_summary.json`
- `output/hybrid_pmcclip_biomedclip_with_distillation—big-distloss/cv_summary.json`
- `output/pmcclip_contrastive/cv_summary.json`
- `output/pmcclip_contrastive_focal/cv_summary.json`
- `output/pmcclip_full/cv_summary.json`
- `output/resnet18_biomedclip_cv/cv_summary.json`
- `output/resnet18_lsal/cv_summary.json`
- `output/resnet18_lsal_new/cv_summary.json`
- `output/resnet50_pmcclip_biomedclip/cv_summary.json`

## ablation（21）

- `checkpoints/ablation_study/ablation_clip_no_doctor_text/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_clip_no_weighted_sampling/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_clip_only/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_clip_short_doctor_text/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_clip_supcon/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_clip_with_doctor_text/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_clip_with_weighted_sampling/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_supcon_clip/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_supcon_clip_class/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_supcon_focal/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/ablation_study/ablation_supcon_only/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_1_3/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_1_5/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_1_7/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_1_9/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_3_5/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_3_7/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_3_9/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_5_7/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_5_9/cv_summary.json`
- `checkpoints/cross_block_liter/ablation_study/starnet_sk_7_9/cv_summary.json`

## clip（15）

- `checkpoints/clip_agent_v2/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/clip_models/resnet18/cv_summary.json`
- `checkpoints/clip_models/resnet18_biomedclip_text/resnet18_biomedclip_text/cv_summary.json`
- `checkpoints/clip_models/resnet18_clip:ViT-B/32/cv_summary.json`
- `checkpoints/clip_models/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/clip_models/resnet18_clip_ViT-B_32/resnet18_clip:ViT-B/32/cv_summary.json`
- `checkpoints/clip_models/resnet18_clip_ViT-B_32/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/clip_models/resnet50_clip_clip_ViT-B_32/resnet50_clip_clip_ViT-B_32/cv_summary.json`
- `checkpoints/clip_models/resnet50_pmcclip_biomedclip_text/resnet50:pmcclip_biomedclip_text/cv_summary.json`
- `checkpoints/clip_models/starnet_s1/cv_summary.json`
- `checkpoints/comparison_experiments/comparison_baseline_clip/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/comparison_experiments/comparison_supcon_clip/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/comparison_experiments/comparison_supcon_clip_class/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/comparison_experiments/comparison_supcon_only/resnet18_clip_ViT-B_32/cv_summary.json`
- `checkpoints/comparison_experiments/comparison_superclip/resnet18_clip_ViT-B_32/cv_summary.json`

## new_data（36）

- `checkpoints/new_data_models/ab_all/casgnet_ab_grn/cv_summary.json`
- `checkpoints/new_data_models/ab_all/casgnet_ab_grn_skunit/cv_summary.json`
- `checkpoints/new_data_models/ab_all/casgnet_ab_sa_grn/cv_summary.json`
- `checkpoints/new_data_models/ab_all/casgnet_ab_sa_skunit/cv_summary.json`
- `checkpoints/new_data_models/ab_all/casgnet_ab_skunit/cv_summary.json`
- `checkpoints/new_data_models/ab_oom/casgnet_ab_sa/cv_summary.json`
- `checkpoints/new_data_models/ab_oom/casgnet_sa_s3/cv_summary.json`
- `checkpoints/new_data_models/ab_oom/casgnet_sa_s4/cv_summary.json`
- `checkpoints/new_data_models/ab_oom/casgnet_sk19/cv_summary.json`
- `checkpoints/new_data_models/casgnet/cv_summary.json`
- `checkpoints/new_data_models/densenet121/cv_summary.json`
- `checkpoints/new_data_models/googlenet/cv_summary.json`
- `checkpoints/new_data_models/lsnet_b/cv_summary.json`
- `checkpoints/new_data_models/mobilenetv4_conv_medium/cv_summary.json`
- `checkpoints/new_data_models/new_casgnet/casgnet/cv_summary.json`
- `checkpoints/new_data_models/resnet18/cv_summary.json`
- `checkpoints/new_data_models/resnet50/cv_summary.json`
- `checkpoints/new_data_models/sa_ablation/casgnet_ab_attn_cbam/cv_summary.json`
- `checkpoints/new_data_models/sa_ablation/casgnet_ab_attn_channel/cv_summary.json`
- `checkpoints/new_data_models/sa_ablation/casgnet_ab_attn_spatial/cv_summary.json`
- `checkpoints/new_data_models/sa_ablation/casgnet_sa_s1/cv_summary.json`
- `checkpoints/new_data_models/sa_ablation/casgnet_sa_s2/cv_summary.json`
- `checkpoints/new_data_models/sk_pos_ablation/casgnet_ab_skpos_all/cv_summary.json`
- `checkpoints/new_data_models/sk_pos_ablation/casgnet_ab_skpos_last1/cv_summary.json`
- `checkpoints/new_data_models/sk_pos_ablation/casgnet_ab_skpos_last2/cv_summary.json`
- `checkpoints/new_data_models/sk_pos_ablation/casgnet_ab_skpos_last3/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk13/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk15/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk17/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk35/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk37/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk39/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk57/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk59/cv_summary.json`
- `checkpoints/new_data_models/sk_size_ablation/casgnet_sk79/cv_summary.json`
- `checkpoints/new_data_models/starnet_s1/cv_summary.json`

## final_*（final_models / final_starnet_models）（61）

- `checkpoints/final_models/ablation/starnet_s1/cv_summary.json`
- `checkpoints/final_models/ablation/starnet_s2/cv_summary.json`
- `checkpoints/final_models/ablation/starnet_s3/cv_summary.json`
- `checkpoints/final_models/ablation/starnet_s4/cv_summary.json`
- `checkpoints/final_models/ablation/starnet_sk_ablation_all/cv_summary.json`
- `checkpoints/final_models/cbam_ablation/starnet_s1_ca/cv_summary.json`
- `checkpoints/final_models/cbam_ablation/starnet_s1_cbam/cv_summary.json`
- `checkpoints/final_models/cbam_ablation/starnet_s1_parallel_attn/cv_summary.json`
- `checkpoints/final_models/cbam_ablation/starnet_s1_sa/cv_summary.json`
- `checkpoints/final_models/sa_ablation/starnet_sa_s2/cv_summary.json`
- `checkpoints/final_models/sa_ablation/starnet_sa_s3/cv_summary.json`
- `checkpoints/final_models/sa_ablation/starnet_sa_s4/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_s1/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_s1_all_grn/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_s1_all_sa/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_s1_grn_only/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_s1_sa_only/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_s1_sk39/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_sk_ablation_all/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_sk_ablation_last1/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_sk_ablation_last2/cv_summary.json`
- `checkpoints/final_models/sk_ablation/starnet_sk_ablation_last3/cv_summary.json`
- `checkpoints/final_models/sk_ablation_all/starnet_s1/cv_summary.json`
- `checkpoints/final_models/sk_ablation_all/starnet_sk_ablation_all/cv_summary.json`
- `checkpoints/final_models/sk_ablation_all/starnet_sk_ablation_last1/cv_summary.json`
- `checkpoints/final_models/sk_ablation_all/starnet_sk_ablation_last2/cv_summary.json`
- `checkpoints/final_models/sk_ablation_all/starnet_sk_ablation_last3/cv_summary.json`
- `checkpoints/final_models/sk_att_grn_ablation/starnet_s1_sa/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk13/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk15/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk17/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk19/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk35/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk37/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk39/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk57/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk59/cv_summary.json`
- `checkpoints/final_models/sk_size_ablation/starnet_s1_sk79/cv_summary.json`
- `checkpoints/final_models/sk_starnet_s1/cv_summary.json`
- `checkpoints/final_models/starnet_s1_final/cv_summary.json`
- `checkpoints/final_models/starnet_sa_s1/cv_summary.json`
- `checkpoints/final_models/starnet_sk_attn_s1/cv_summary.json`
- `checkpoints/final_models/starnet_sk_s1/cv_summary.json`
- `checkpoints/final_starnet_models/attention_ablation/starnet_s1/cv_summary.json`
- `checkpoints/final_starnet_models/attention_ablation/starnet_s2/cv_summary.json`
- `checkpoints/final_starnet_models/attention_ablation/starnet_s3/cv_summary.json`
- `checkpoints/final_starnet_models/attention_ablation/starnet_s4/cv_summary.json`
- `checkpoints/final_starnet_models/cross_attention/starnet_s1_final/cv_summary.json`
- `checkpoints/final_starnet_models/cross_grn/starnet_sa_s1/cv_summary.json`
- `checkpoints/final_starnet_models/early_stop_ab/starnet_s1/cv_summary.json`
- `checkpoints/final_starnet_models/final_model/starnet_s1/cv_summary.json`
- `checkpoints/final_starnet_models/final_model/starnet_s1_final/cv_summary.json`
- `checkpoints/final_starnet_models/final_model/starnet_sa_s1/cv_summary.json`
- `checkpoints/final_starnet_models/gln_attention/starnet_sa_s1/cv_summary.json`
- `checkpoints/final_starnet_models/sa_variants/starnet_sa_s1/cv_summary.json`
- `checkpoints/final_starnet_models/starnet_s1/cv_summary.json`
- `checkpoints/final_starnet_models/starnet_s1_cross_star/cv_summary.json`
- `checkpoints/final_starnet_models/starnet_s1_cross_star_add/cv_summary.json`
- `checkpoints/final_starnet_models/starnet_s1_cross_star_samescale/cv_summary.json`
- `checkpoints/final_starnet_models/starnet_s2/cv_summary.json`
- `checkpoints/final_starnet_models/starnet_s3/cv_summary.json`

## other（18）

- `checkpoints/cross_block_liter/starnet_s1_final/cv_summary.json`
- `checkpoints/cross_block_liter/starnet_sa_s1/cv_summary.json`
- `checkpoints/cv_multi_models/convnextv2_base/cv_summary.json`
- `checkpoints/cv_multi_models/convnextv2_tiny/cv_summary.json`
- `checkpoints/cv_multi_models/mambaout_tiny/cv_summary.json`
- `checkpoints/cv_multi_models/mobilenetv4_conv_medium/cv_summary.json`
- `checkpoints/cv_multi_models/mobilenetv4_conv_small/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_artifact_s1/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_cf_s3/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_gated_s1/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_s1_cross_star/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_s1_final/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_s1_grn/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_s2_final/cv_summary.json`
- `checkpoints/cv_multi_models/starnet_s3_final/cv_summary.json`
- `checkpoints/lightsk_models/lightsk/cv_summary.json`
- `checkpoints/lightsk_models/lightsk_base/cv_summary.json`
- `checkpoints/lightsk_models/lightsk_small/cv_summary.json`
