import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cross_star_block():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(-1, 23)
    ax.set_ylim(-5, 5)
    ax.axis('off')

    # Styles mimicking the reference image
    colors = {
        'input_green': '#A9D18E',
        'att_blue': '#9DC3E6',   # Spat-Att
        'dw_yellow': '#FFE699',  # DW-Conv
        'fc_orange': '#F4B183',  # FC / Conv
        'grn_blue': '#BDD7EE',   # GRN
        'op_gray': '#D9D9D9',    # Mul/Concat
        'text': '#000000'
    }
    
    # Helper to draw box
    def draw_box(x, y, w, h, color, label, fontsize=10, rotation=90):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='white', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize, 
                rotation=rotation, color=colors['text'], fontweight='bold')
        return (x + w, y + h/2) # Return output connection point

    # Helper to draw circle (operations)
    def draw_circle(x, y, radius, label):
        circle = patches.Circle((x, y), radius, linewidth=1.5, edgecolor='gray', facecolor=colors['op_gray'])
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=16, fontweight='bold')
        return (x, y)

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

    # --- Main Pipeline ---

    # 1. Input
    draw_box(0, -1.5, 1, 3, colors['input_green'], "", rotation=0)
    ax.text(0.5, -2, "Input", ha='center', fontsize=10)
    
    # Residual Line Start
    ax.plot([0.5, 0.5, 21.5, 21.5], [1.5, 3.5, 3.5, 1.5], color='black', lw=1.5)
    ax.arrow(21.5, 2.0, 0, -0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # Connection to Spat-Att
    draw_arrow(1, 0, 2, 0)

    # 2. Spat-Att
    out_sa = draw_box(2, -1.5, 1.5, 3, colors['att_blue'], "Spat-Att", rotation=90)
    
    # Connection
    draw_arrow(out_sa[0], 0, 4.5, 0)

    # 3. DW-Conv 1
    out_dw1 = draw_box(4.5, -1.5, 1.5, 3, colors['dw_yellow'], "DW-Conv\n(7x7)", rotation=90)

    # --- Branching (The Cross Star) ---
    
    # Split point
    split_x = 6.5
    draw_arrow(out_dw1[0], 0, split_x, 0)
    
    # Branch Y12 (Top) components
    # 3x3 (A)
    draw_arrow(split_x, 0, 7.5, 2)
    out_3a = draw_box(7.5, 1.2, 1.2, 1.6, colors['fc_orange'], "Conv\n3x3", rotation=0)
    # 7x7 (B)
    draw_arrow(split_x, 0, 7.5, 0.5)
    out_7b = draw_box(7.5, -0.3, 1.2, 1.6, colors['fc_orange'], "Conv\n7x7", rotation=0)
    
    # Branch Y21 (Bottom) components
    # 7x7 (A)
    draw_arrow(split_x, 0, 7.5, -1.3)
    out_7a = draw_box(7.5, -2.1, 1.2, 1.6, colors['fc_orange'], "Conv\n7x7", rotation=0)
    # 3x3 (B)
    draw_arrow(split_x, 0, 7.5, -2.8)
    out_3b = draw_box(7.5, -3.6, 1.2, 1.6, colors['fc_orange'], "Conv\n3x3", rotation=0)

    # Multiplications
    # Mul Top
    mul_top = draw_circle(10, 1.25, 0.4, "*")
    draw_arrow(out_3a[0], 2, 9.6, 1.4) # slight adjust for visual
    draw_arrow(out_7b[0], 0.5, 9.6, 1.1)

    # Mul Bottom
    mul_bot = draw_circle(10, -1.25, 0.4, "*")
    draw_arrow(out_7a[0], -1.3, 9.6, -1.1)
    draw_arrow(out_3b[0], -2.8, 9.6, -1.4)
    
    # GRNs
    # GRN Top
    draw_arrow(mul_top[0]+0.4, mul_top[1], 11.5, mul_top[1])
    out_grn1 = draw_box(11.5, 0.25, 1.2, 2, colors['grn_blue'], "GRN", rotation=90)
    
    # GRN Bot
    draw_arrow(mul_bot[0]+0.4, mul_bot[1], 11.5, mul_bot[1])
    out_grn2 = draw_box(11.5, -2.25, 1.2, 2, colors['grn_blue'], "GRN", rotation=90)

    # --- Fusion ---
    
    # Concat
    cat_node = draw_circle(14, 0, 0.5, "C")
    # Arrows to concat
    draw_arrow(out_grn1[0], 1.25, 13.6, 0.2)
    draw_arrow(out_grn2[0], -1.25, 13.6, -0.2)
    
    # FC (Projection g)
    draw_arrow(cat_node[0]+0.5, 0, 15.5, 0)
    out_fc = draw_box(15.5, -1.5, 1.2, 3, colors['fc_orange'], "FC\n(1x1)", rotation=90)
    
    # DW-Conv 2 (Final projection)
    draw_arrow(out_fc[0], 0, 17.5, 0)
    out_dw2 = draw_box(17.5, -1.5, 1.5, 3, colors['dw_yellow'], "DW-Conv\n(7x7)", rotation=90)

    # Output
    draw_arrow(out_dw2[0], 0, 21, 0)
    draw_box(21, -1.5, 1, 3, colors['input_green'], "", rotation=0)
    ax.text(21.5, -2, "Output", ha='center', fontsize=10)

    # Title
    ax.text(11, -4.5, "CrossStar Block", fontsize=22, fontweight='bold', ha='center')

    # Annotations for branches
    ax.text(7.0, 2.5, "Branch 1 (Local mod Context)", fontsize=9, color='gray')
    ax.text(7.0, -4.0, "Branch 2 (Context mod Local)", fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig('cross_star_block.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    return fig

# 检查是否已安装必要的库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    print("库已安装，开始生成图像...")
except ImportError:
    print("正在安装必要的库...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    print("库安装完成，开始生成图像...")

# Generate the plot
fig = draw_cross_star_block()
print("图像已保存为 cross_star_block.png")
