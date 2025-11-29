import matplotlib.pyplot as plt
import numpy as np


def draw(latex):
    # 配置字体（确保中文、负号显示正常）
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False
    # 开启 LaTeX 渲染（用于正确显示花体等特殊符号）
    fontsize = 36
    if latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"  # 加载数学公式相关包
        fontsize = 20
    label_fontsize = 40

    data_groups = {
        r"$\mathcal{L}$-$\text{all}$": (0.379, 0.393),
        r"$\mathcal{L}$-$\text{kl}$": (0.380, 0.393),
        r"$\mathcal{L}$-$\text{hmm}$": (0.373, 0.385),
        r"$\mathcal{L}$-$\text{rec}$": (0.379, 0.393),  # 修复了LaTeX语法错误
        # r"$\text{Only L1}$": (0.544, 0.396, 0.341, 0.568),
    }

    # 指标名、分组名、颜色映射（按分组分配颜色，保证不同setting颜色固定）
    metrics = ["MSE", "MAE"]
    groups = list(data_groups.keys())
    # 为每个分组分配颜色（与原代码指标颜色顺序对应，保持视觉一致）
    # group_colors = ['#FFF3CE', '#FAD8D4', '#D9E8FC', '#BAC8D3', '#D6E9D5']
    # group_colors = ['#90BFD5', '#FCBB44', '#A0D292', '#F1766D', '#7A70B5']
    group_colors = ['#F1766D', '#FCBB44', '#839DD1', '#7A70B5', '#A0D292']

    fig, axes = plt.subplots(1, 2, figsize=(32, 8))  # 修改了子图数量为4

    # 为每个指标创建单独的柱状图
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # 为当前指标，按分组循环绘图（每个分组对应一种颜色）
        for j, group in enumerate(groups):
            value = data_groups[group][i]  # 提取当前分组、当前指标的值
            # 绘制柱状图，用分组对应的颜色
            rect = ax.bar(j, value, width=0.6, color=group_colors[j], label=group)

            # 添加数值标签
            ax.annotate(
                f'{value:.3f}',
                xy=(j, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=fontsize
            )

        # 子图配置
        ax.set_title(metric, fontsize=label_fontsize)  # 指标名作为子图标题
        ax.set_xticks(np.arange(len(groups)))  # x轴刻度对应分组
        ax.set_xticklabels(groups, fontsize=fontsize, rotation=15, ha='right')  # 分组名作为x轴标签，增加旋转角度和对齐方式
        ax.tick_params(axis='y', labelsize=fontsize)  # 设置Y轴刻度字体大小
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # y轴网格线
        # 设置y轴范围（基于当前指标的所有分组值）
        all_values = [data_groups[g][i] for g in groups]
        ax.set_ylim(bottom=min(all_values) * 0.95, top=max(all_values) * 1.05)
        if i == 0:
            ax.set_ylabel("Values", fontsize=label_fontsize)  # 设置 y 轴标签及字体大小

    # 调整子图间距，增加wspace值
    plt.subplots_adjust(wspace=2)
    # 保存和显示图形
    plt.tight_layout()
    plt.savefig('ablation_loss_etth.pdf', format='pdf', dpi=640, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    is_latex = False  # 设置为True启用LaTeX渲染
    draw(is_latex)