# 实验配置追踪表

本文件用于追踪所有实验配置的参数和运行状态。

> **注意**: 请只修改「运行机器」、「运行状态」和「说明」列，其他列会在重新生成时自动更新。

| Group  | ppo epoch | Strategy        | k_pos | λ_pos | Peak | k_neg | λ_neg | 运行机器 | 运行状态   | 说明                                 |
|:-------|:----------|:----------------|------:|------:|-----:|------:|------:|:-----|:-------|:-----------------------------------|
| G1     | 16        | Mimic           |   1.0 |   0.8 | 1.25 |   3.0 |   1.5 | 1.5  | ISv0-1 | XX                                 |
| G2     | 16        | Mimic           |   1.2 |   1.0 |  1.2 |   5.0 |   2.5 | 2.5  | ISv0-2 | XXXXXX                             |
| G3     | 16        | Mimic           |   1.5 |   1.2 | 1.25 |   3.0 |   3.0 | 3.0  | ISv0-3 | X                                  |
| G4     | 16        | Legacy          |   2.0 |   1.6 | 1.25 |   4.0 |   4.0 | 4.0  | ISv0-4 | X                                  |
| G5     | 16        | Math            |   2.0 |   2.0 |  1.0 |   4.0 |   2.5 | 2.5  | ISv0-5 | XX                                 |
| G6     | 16        | Math            |   1.5 |   1.5 |  1.0 |   4.0 |   2.5 | 2.5  | ISv0-6 | XX                                 |
| G7     | 16        | Math            |   3.0 |   3.0 |  1.0 |   4.0 |   2.5 | 2.5  | ISv0-7 | XX                                 |
| G8     | 16        | Math            |   4.0 |   4.0 |  1.0 |   4.0 |   2.5 | 2.5  | ISv0-8 | XX                                 |
| G9     | 16        | Drift           |   2.0 |   2.2 |  0.9 |   4.0 |   3.0 | 3.0  | ISv1-1 | XXX                                |
| G10    | 16        | Drift           |   2.0 |   2.5 |  0.8 |   5.0 |   4.0 | 4.0  | ISv1-2 | XX                                 |
| G11    | 16        | Drift           |   1.0 |  1.25 |  0.8 |   4.0 |   2.5 | 2.5  | ISv1-3 | ⚠️                                 |
| G12    | 16        | Drift           |   1.5 |   2.5 |  0.6 |   4.0 |   2.5 | 2.5  | ISv1-4 | XXX                                |
| G13    | 16        | NegDef          |   2.0 |   2.0 |  1.0 |   3.0 |   1.5 | 1.5  | ISv1-5 | XXXX                               |
| G14    | 16        | NegDef          |   2.0 |   2.0 |  1.0 |   6.0 |   3.0 | 3.0  | ISv1-6 | ⚠️                                 |
| G15    | 16        | NegDef          |   2.0 |   2.0 |  1.0 |   6.0 |   6.0 | 6.0  | ISv1-7 | X                                  |
| G16    | 16        | Wild            |   2.0 |   1.8 | 1.11 |   4.0 |   2.5 | 2.5  | ISv1-8 | XX                                 |
| G17    | 16        | Wild            |   1.5 |   1.0 |  1.5 |   5.0 |   5.0 | 5.0  | IDE-1  | X                                  |
| G18    | 16        | Wild            |   3.0 |   3.0 |  1.0 |   3.0 |   3.0 | 3.0  | IDE-2  | X                                  |
| G19    | 16        | Wild            |   4.0 |   3.2 | 1.25 |   4.0 |   4.0 | 4.0  | PAI-1  | X                                  |
| G20    | 16        | Mimic_V2        |   2.0 |   2.0 |    1 |   8.0 |   4.0 | 4.0  | ISv0-3 |                                    |
| G21    | 16        | Legacy_Fix      |   2.0 |   2.0 |    1 |   6.0 |   4.0 | 4.0  | ISv0-4 | XXXX                               |
| G22    | 16        | Dome_Fix        |   2.0 |   2.0 |  1.0 |   4.0 |   2.0 | 2.0  | ISv1-7 | XXXX                               |
| G23    | 16        | Gamble_Safe     |   2.0 |   2.0 |  1.0 |   3.0 |   2.0 | 2.0  | IDE-1  | XXX                                |
| G24    | 16        | Asym_Prec       |   3.0 |   2.0 |  1.5 |   4.0 |   2.5 | 2.5  | IDE-2  | XXX                                |
| G25    | 16        | Box_Shape       |   4.0 |   2.0 |  2.0 |   8.0 |   4.0 | 4.0  | PAI-1  | XX                                 |
| G26    | 16        | Stable_Tight    |   2.7 |   3.0 |  0.9 |   6.0 |   3.0 | 3.0  | ISv0-1 | XXXXXXX                            |
| G27    | 16        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.0 | 2.0  | ISv0-5 | 宽域惩罚: Pos左移(0.8) + Neg极宽(Peak 2.5) |
| G28    | 16        | Wide_Net        |   2.5 |   2.5 |  1.0 |   5.0 |   2.0 | 2.0  | ISv0-6 | XXXXX                              |
| G29    | 16        | Precise_Wall    |   3.0 |   3.0 |  1.0 |   6.0 |   2.4 | 2.4  | ISv0-7 | XXXXX                              |
| G30    | 16        | Left_Anchor     |   2.5 |   3.2 | 0.78 |   6.0 |   3.0 | 3.0  | ISv0-8 | 强锚定: Pos强力左移限制探索                   |
| G31    | 16        | Conservative    |   2.0 |     3 | 0.67 |   8.0 |   3.2 | 3.2  | ISv1-2 | XXXXX                              |
| G32    | 16        | High_Order_L    |   4.0 |   5.0 |  0.8 |   8.0 |   3.0 | 3.0  | ISv1-8 | XXXXX                              |
| G33    | 16        | Balanced_Fix    |   3.0 |   4.0 | 0.75 |   5.0 |   2.0 | 2.0  | PAI-1  | XXXXXXX                            |
| G34    | 16        | Flat_Punisher   |   2.5 |   3.2 | 0.78 |   2.0 |   0.4 | 0.4  | ISv1-1 | XXXXX                              |
| G35    | 16        | Deep_Guard      |   3.0 |   4.0 | 0.75 |   4.0 |   0.8 | 0.8  | ISv1-4 | XXXXX                              |
| G36    | 16        | Ramp_Wall       |   2.0 |   2.5 |  0.8 |   1.5 |   0.2 | 0.2  | IDE-1  |                                    |
| G37    | 16        | Heavy_Anchor    |   4.0 |   5.0 |  0.8 |   3.0 |   0.5 | 0.5  | IDE-2  | XXXXX                              |
| G38    | 16        | Titan_Guard     |   3.0 |  3.75 |  0.8 |   3.0 |   0.5 | 0.5  | ISv1-5 | XXXXXXX                            |
| G39    | 16        | High_Order_Wall |   2.5 | 3.125 |  0.8 |   6.0 |  0.75 | 0.75 | ISv0-4 | XXXXX                              |
| G40    | 16        | Smooth_Fortress |   2.0 |   2.5 |  0.8 |   4.0 |   0.5 | 0.5  | ISv0-4 | XXXXXXX                            |
| G41    | 16        | G14_Evo         |   3.0 |   4.0 | 0.75 |   6.0 |  2.72 | 2.72 | ISv0-6 |                                    |
| G42    | 16        | Urgent_Teach    |   3.0 |   4.0 | 0.75 |   4.5 |   2.5 | 2.5  | ISv0-7 | XXXXXX                             |
| G43    | 16        | Soft_Explore    |   2.0 |  2.66 | 0.75 |   5.0 |   2.0 | 2.0  | ISv1-1 | ⚠️                                 |
| G44    | 16        | Anchor_Test     |   3.0 |  3.33 |  0.9 |   5.0 |   2.0 | 2.0  | ISv1-2 | ⚠️                                 |
| G45    | 16        | Wide_Net        |   3.0 |   4.0 | 0.75 |   2.5 |  0.68 | 0.68 | ISv1-4 | XXXXXXX                            |
| G46    | 16        | Strict_Anchor   |   5.0 |   5.0 |  0.8 |   5.0 |   2.0 | 2.0  | ISv1-7 | XXXXXXX                            |
| G47    | 16        | G2_Revival      |   2.0 |  0.91 |  1.1 |   5.0 |   2.0 | 2.0  | ISv1-8 | ⚠️                                 |
| G48    | 16        | The_Sniper      |   3.0 |   4.0 | 0.75 |   8.0 |   3.5 | 3.5  | IDE-2  |                                    |
| G33_4  | 4         | PPO_4           |   3.0 |   4.0 | 0.75 |   5.0 |   2.0 | 2.0  | ISv0-2 |                                    |
| G41_4  | 4         | PPO_4           |   3.0 |   4.0 | 0.75 |   6.0 |  2.72 | 2.72 | ISv0-7 |                                    |
| G27_4  | 4         | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.0 | -    | ISv0-1 |                                    |
| G47_4  | 4         | G2_Revival      |   2.0 |  0.91 |  2.2 |   5.0 |   2.0 | -    | ISv0-4 |                                    |
| G49    | 16        | G2_Revival      |   2.0 |   0.8 |  2.0 |   5.0 |   2.0 | -    | ISv0-7 |                                    |
| G50    | 16        | G2_Revival      |   2.0 |   1.0 |  2.0 |   5.0 |   2.0 | -    | ISv1-5 | ⚠️                                 |
| G49_4  | 4         | G2_Revival      |   2.0 |   0.8 |  2.0 |   5.0 |   2.0 | -    | ISv1-7 | ⚠️                                 |
| G50_4  | 4         | G2_Revival      |   2.0 |   1.0 |  2.0 |   5.0 |   2.0 | -    | IDE-3  |                                    |
| G27_64 | 64        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.0 | -    | IDE-4  |                                    |
|        | 0         |                 |     0 |     0 |    0 |     0 |     0 | -    | -      |                                    |
| X1_4   | 4         | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.0 | -    | ISv0-1 |                                    |
| X2_4   | 4         | Broad_Punish    |   2.0 |   2.5 |  0.8 |   4.0 |   2.0 | -    | ISv0-2 | X                                  |
| X3_4   | 4         | Broad_Punish    |   2.0 |   2.5 |  0.8 |   6.0 |   2.0 | -    | ISv0-3 |                                    |
| X4_4   | 4         | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.5 | -    | ISv0-4 | X                                  |
| Y1_4   | 4         | G2_Revival      |   2.0 |  0.91 |  2.2 |   5.0 |   2.0 | -    | ISv0-5 |                                    |
| Y2_4   | 4         | G2_Revival      |   2.0 |   1.2 |  2.2 |   5.0 |   2.0 | -    | ISv0-6 |                                    |
| Y3_4   | 4         | G2_Revival      |   2.0 |  0.81 |  2.2 |   5.0 |   2.0 | -    | ISv0-7 | re-run                             |
| Y4_4   | 4         | G2_Revival      |   1.5 |  0.91 |  2.2 |   5.0 |   2.0 | -    | ISv0-8 |                                    |
| X1_16  | 16        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.0 | -    | ISv1-1 | XX                                 |
| X2_16  | 16        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   4.0 |   2.0 | -    | ISv1-2 | X                                  |
| X3_16  | 16        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   6.0 |   2.0 | -    | ISv1-3 |                                    |
| X4_16  | 16        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.5 | -    | ISv1-4 | X                                  |
| Y1_16  | 16        | G2_Revival      |   2.0 |  0.91 |  2.2 |   5.0 |   2.0 | -    | ISv1-5 |                                    |
| Y2_16  | 16        | G2_Revival      |   2.0 |   1.2 |  2.2 |   5.0 |   2.0 | -    | ISv1-6 |                                    |
| Y3_16  | 16        | G2_Revival      |   2.0 |  0.81 |  2.2 |   5.0 |   2.0 | -    | ISv1-7 | XX -> Z1_dynamic                   |
| Y4_16  | 16        | G2_Revival      |   1.5 |  0.91 |  2.2 |   5.0 |   2.0 | -    | ISv1-8 |                                    |
| X1_64  | 64        | Broad_Punish    |   2.0 |   2.5 |  0.8 |   5.0 |   2.0 | -    | ISv2-1 | X                                  |
| Y1_64  | 64        | G2_Revival      |   2.0 |  0.91 |  2.2 |   5.0 |   2.0 | -    | ISv2-2 | X                                  |
| Y5_4   | 4         | G2_Revival      |   2.0 |   1.5 |  1.3 |   5.0 |   2.0 | -    | ISv0-2 |                                    |
| Y6_4   | 4         | G2_Revival      |   2.0 |   1.0 |  2.0 |   6.0 |   2.0 | -    | ISv0-4 |                                    |
| Y5_16  | 16        | G2_Revival      |   2.0 |   1.5 |  1.3 |   5.0 |   2.0 | -    | ISv1-2 |                                    |
| Y6_16  | 16        | G2_Revival      |   2.0 |  0.91 |  2.2 |   6.0 |   2.0 | -    | ISv1-4 |                                    |
| Y7_16  | 16        | G2_Revival      |   2.0 |  0.91 |  2.2 |   8.0 |   2.0 | -    | ISv2-1 |                                    |
| Y8_16  | 16        | G2_Revival      |   2.0 |   1.5 |  2.0 |   6.0 |   2.0 | -    | ISv2-2 |                                    |
| Z1_16  | 16        | G2_Revival      |   1.5 |  0.91 | 1.64 |   4.0 |   1.0 | -    | ISv1-1 |                                    |

---

## 运行状态说明

- `-`: 未设置
- `待运行`: 等待执行
- `运行中`: 正在执行
- `已完成`: 执行完成
- `失败`: 执行失败
- `暂停`: 暂时停止

## 配置参考（仅供参考）

以下是各组配置的原始说明，供参考使用：

| Group | 原始说明 |
|:------|:---------|
| G1 | 模拟 PPO 线性+宽容 |
| G2 | 模拟 PPO + 强 Lower Clip |
| G3 | 模拟 PPO + 强 Upper Clip |
| G4 | 前冠军复刻 + 强防 |
| G5 | 标准推荐 (Bell) |
| G6 | 宽钟形 |
| G7 | 尖钟形 |
| G8 | 针尖 (高精度) |
| G9 | 微左偏 |
| G10 | 强左偏纠偏 |
| G11 | 线性左偏 |
| G12 | 极端深锚 |
| G13 | 负向极宽容 (PPO [0.8,3] 模拟) |
| G14 | 负向只杀小 w |
| G15 | 负向全杀 (铁穹) |
| G16 | 微右偏激进 |
| G17 | 赌徒模式 (High Risk) |
| G18 | 完全对称 |
| G19 | 模拟 PPO 的'平顶'感 |
| G20 |  |
| G21 |  |
| G22 |  |
| G23 |  |
| G24 |  |
| G25 |  |
| G26 | G14改良: Pos微左移收紧 + Neg标准铁壁 |
| G27 | 宽域惩罚: Pos左移(0.8) + Neg极宽(Peak 2.5) |
| G28 | 广撒网: Neg覆盖极宽范围防止后期逃逸 |
| G29 | 高精铁壁: Pos高精(k=3) + Neg高阶宽域 |
| G30 | 强锚定: Pos强力左移限制探索 |
| G31 | 极保守: Pos深锚 + Neg究极左侧截断 |
| G32 | 高阶左偏: 模拟非对称盒式约束 |
| G33 | 综合修正: 高精左移Pos + 宽域Neg |
| G34 |  |
| G35 |  |
| G36 |  |
| G37 |  |
| G38 |  |
| G39 |  |
| G40 |  |
| G41 |  |
| G42 |  |
| G43 |  |
| G44 |  |
| G45 |  |
| G46 |  |
| G47 |  |
| G48 |  |
| G33_4 |  |
| G41_4 |  |
| G27_4 |  |
| G47_4 |  |
| G49 |  |
| G50 |  |
| G49_4 |  |
| G50_4 |  |
| G27_64 |  |
|   |  |
| X1_4 |  |
| X2_4 |  |
| X3_4 |  |
| X4_4 |  |
| Y1_4 |  |
| Y2_4 |  |
| Y3_4 |  |
| Y4_4 |  |
| X1_16 |  |
| X2_16 |  |
| X3_16 |  |
| X4_16 |  |
| Y1_16 |  |
| Y2_16 |  |
| Y3_16 |  |
| Y4_16 |  |
| X1_64 |  |
| Y1_64 |  |
| Y5_4 |  |
| Y6_4 |  |
| Y5_16 |  |
| Y6_16 |  |
| Y7_16 |  |
| Y8_16 |  |
| Z1_16 |  |

## 快速命令参考

```bash
# G1 - 模拟 PPO 线性+宽容
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.0_0.8_3.0_1.5_G1.sh

# G2 - 模拟 PPO + 强 Lower Clip
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.2_1.0_5.0_2.5_G2.sh

# G3 - 模拟 PPO + 强 Upper Clip
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.5_1.2_3.0_3.0_G3.sh

# G4 - 前冠军复刻 + 强防
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_1.6_4.0_4.0_G4.sh

# G5 - 标准推荐 (Bell)
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_4.0_2.5_G5.sh

# G6 - 宽钟形
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.5_1.5_4.0_2.5_G6.sh

# G7 - 尖钟形
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_3.0_4.0_2.5_G7.sh

# G8 - 针尖 (高精度)
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_4.0_4.0_4.0_2.5_G8.sh

# G9 - 微左偏
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.2_4.0_3.0_G9.sh

# G10 - 强左偏纠偏
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_5.0_4.0_G10.sh

# G11 - 线性左偏
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.0_1.25_4.0_2.5_G11.sh

# G12 - 极端深锚
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.5_2.5_4.0_2.5_G12.sh

# G13 - 负向极宽容 (PPO [0.8,3] 模拟)
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_3.0_1.5_G13.sh

# G14 - 负向只杀小 w
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_6.0_3.0_G14.sh

# G15 - 负向全杀 (铁穹)
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_6.0_6.0_G15.sh

# G16 - 微右偏激进
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_1.8_4.0_2.5_G16.sh

# G17 - 赌徒模式 (High Risk)
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.5_1.0_5.0_5.0_G17.sh

# G18 - 完全对称
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_3.0_3.0_3.0_G18.sh

# G19 - 模拟 PPO 的'平顶'感
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_4.0_3.2_4.0_4.0_G19.sh

# G20 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_8.0_4.0_G20.sh

# G21 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_6.0_4.0_G21.sh

# G22 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_4.0_2.0_G22.sh

# G23 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_3.0_2.0_G23.sh

# G24 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_2.0_4.0_2.5_G24.sh

# G25 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_4.0_2.0_8.0_4.0_G25.sh

# G26 - G14改良: Pos微左移收紧 + Neg标准铁壁
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.7_3.0_6.0_3.0_G26.sh

# G27 - 宽域惩罚: Pos左移(0.8) + Neg极宽(Peak 2.5)
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_5.0_2.0_G27.sh

# G28 - 广撒网: Neg覆盖极宽范围防止后期逃逸
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.5_2.5_5.0_2.0_G28.sh

# G29 - 高精铁壁: Pos高精(k=3) + Neg高阶宽域
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_3.0_6.0_2.4_G29.sh

# G30 - 强锚定: Pos强力左移限制探索
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.5_3.2_6.0_3.0_G30.sh

# G31 - 极保守: Pos深锚 + Neg究极左侧截断
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_3_8.0_3.2_G31.sh

# G32 - 高阶左偏: 模拟非对称盒式约束
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_4.0_5.0_8.0_3.0_G32.sh

# G33 - 综合修正: 高精左移Pos + 宽域Neg
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_4.0_5.0_2.0_G33.sh

# G34 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.5_3.2_2.0_0.4_G34.sh

# G35 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_4.0_4.0_0.8_G35.sh

# G36 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_1.5_0.2_G36.sh

# G37 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_4.0_5.0_3.0_0.5_G37.sh

# G38 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_3.75_3.0_0.5_G38.sh

# G39 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.5_3.125_6.0_0.75_G39.sh

# G40 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_4.0_0.5_G40.sh

# G41 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_4.0_6.0_2.72_G41.sh

# G42 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_4.0_4.5_2.5_G42.sh

# G43 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.66_5.0_2.0_G43.sh

# G44 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_3.33_5.0_2.0_G44.sh

# G45 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_4.0_2.5_0.68_G45.sh

# G46 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_5.0_5.0_5.0_2.0_G46.sh

# G47 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_0.91_5.0_2.0_G47.sh

# G48 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_3.0_4.0_8.0_3.5_G48.sh

# G33_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_3.0_4.0_5.0_2.0_G33_4.sh

# G41_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_3.0_4.0_6.0_2.72_G41_4.sh

# G27_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_2.5_5.0_2.0_G27_4.sh

# G47_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_0.91_5.0_2.0_G47_4.sh

# G49 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_0.8_5.0_2.0_G49.sh

# G50 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_1.0_5.0_2.0_G50.sh

# G49_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_0.8_5.0_2.0_G49_4.sh

# G50_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_1.0_5.0_2.0_G50_4.sh

# G27_64 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_64_2.0_2.5_5.0_2.0_G27_64.sh

#   - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_0_0_0_0_0_ .sh

# X1_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_2.5_5.0_2.0_X1_4.sh

# X2_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_2.5_4.0_2.0_X2_4.sh

# X3_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_2.5_6.0_2.0_X3_4.sh

# X4_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_2.5_5.0_2.5_X4_4.sh

# Y1_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_0.91_5.0_2.0_Y1_4.sh

# Y2_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_1.2_5.0_2.0_Y2_4.sh

# Y3_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_0.81_5.0_2.0_Y3_4.sh

# Y4_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_1.5_0.91_5.0_2.0_Y4_4.sh

# X1_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_5.0_2.0_X1_16.sh

# X2_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_4.0_2.0_X2_16.sh

# X3_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_6.0_2.0_X3_16.sh

# X4_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.5_5.0_2.5_X4_16.sh

# Y1_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_0.91_5.0_2.0_Y1_16.sh

# Y2_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_1.2_5.0_2.0_Y2_16.sh

# Y3_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_0.81_5.0_2.0_Y3_16.sh

# Y4_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.5_0.91_5.0_2.0_Y4_16.sh

# X1_64 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_64_2.0_2.5_5.0_2.0_X1_64.sh

# Y1_64 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_64_2.0_0.91_5.0_2.0_Y1_64.sh

# Y5_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_1.5_5.0_2.0_Y5_4.sh

# Y6_4 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_4_2.0_1.0_6.0_2.0_Y6_4.sh

# Y5_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_1.5_5.0_2.0_Y5_16.sh

# Y6_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_0.91_6.0_2.0_Y6_16.sh

# Y7_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_0.91_8.0_2.0_Y7_16.sh

# Y8_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_1.5_6.0_2.0_Y8_16.sh

# Z1_16 - 
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.5_0.91_4.0_1.0_Z1_16.sh

```
