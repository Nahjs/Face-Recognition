import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# 为matplotlib设置中文字体，以防需要显示中文
# 假设SimHei字体已安装在系统上，如果未安装，可视化中文可能仍有问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # 添加一个通用字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def visualize_acceptance_results(csv_path="acceptance_results.csv", output_dir="acceptance_plots"):
    """
    读取验收结果CSV，生成可视化图表。
    """
    if not os.path.exists(csv_path):
        print(f"错误：未找到结果文件 {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条验收结果。")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 确保准确率是数值类型，处理可能的非数值条目 (如 N/A, ERROR)
    df['Acceptance_Accuracy'] = pd.to_numeric(df['Acceptance_Accuracy'], errors='coerce')
    df_valid = df.dropna(subset=['Acceptance_Accuracy'])

    if df_valid.empty:
        print("没有有效的准确率数据可供可视化。")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 示例可视化：不同损失类型的平均准确率
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Loss_Type', y='Acceptance_Accuracy', data=df_valid)
    plt.title('平均验收准确率 (按损失类型)')
    plt.ylabel('准确率')
    plt.xlabel('损失类型')
    plt.ylim(0, 1.0) # 准确率范围0-1
    plot_path = os.path.join(output_dir, 'avg_accuracy_by_loss_type.png')
    plt.savefig(plot_path)
    plt.close() # 关闭图表，释放内存
    print(f"图表已保存到 {plot_path}")

    # 示例可视化：按实验名称（完整配置组合）排序的准确率
    # 可以按准确率降序排序以便找到最佳配置
    df_sorted = df_valid.sort_values(by='Acceptance_Accuracy', ascending=False).copy() # 使用 .copy() 避免 SettingWithCopyWarning
    df_sorted['Experiment_Name'] = df_sorted['Combo_Name'] # 使用 Combo_Name 作为 Experiment_Name
    plt.figure(figsize=(12, max(6, len(df_sorted) * 0.5))) # 根据条目数调整图表高度
    sns.barplot(x='Acceptance_Accuracy', y='Experiment_Name', data=df_sorted)
    plt.title('验收准确率 (按配置)')
    plt.xlabel('准确率')
    plt.ylabel('配置名称')
    plt.xlim(0, 1.0)
    plt.tight_layout() # 自动调整布局以防止标签重叠
    plot_path = os.path.join(output_dir, 'accuracy_by_config.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"图表已保存到 {plot_path}")

    # --- 添加更多验收结果可视化图表 ---

    # 1. 按骨干网络 (Model_Type) 对比平均准确率
    if 'Model_Type' in df_valid.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model_Type', y='Acceptance_Accuracy', data=df_valid)
        plt.title('平均验收准确率 (按骨干网络类型)')
        plt.ylabel('准确率')
        plt.xlabel('骨干网络类型')
        plt.ylim(0, 1.0)
        plot_path = os.path.join(output_dir, 'avg_accuracy_by_model_type.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"图表已保存到 {plot_path}")

    # 2. 按优化器 (Optimizer_Type) 对比平均准确率
    if 'Optimizer_Type' in df_valid.columns: # 注意：CSV中没有Optimizer_Type，这里假设可以从Combo_Name提取或添加
        # 尝试从 Combo_Name 提取 Optimizer_Type
        df_valid['Optimizer_Type'] = df_valid['Combo_Name'].apply(lambda x: x.split('__')[2] if len(x.split('__')) > 2 else 'Unknown')
        if not df_valid['Optimizer_Type'].empty and not all(df_valid['Optimizer_Type'] == 'Unknown'):
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Optimizer_Type', y='Acceptance_Accuracy', data=df_valid)
            plt.title('平均验收准确率 (按优化器)')
            plt.ylabel('准确率')
            plt.xlabel('优化器')
            plt.ylim(0, 1.0)
            plot_path = os.path.join(output_dir, 'avg_accuracy_by_optimizer.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"图表已保存到 {plot_path}")
        else:
            print("警告: 无法从 Combo_Name 中提取有效的 Optimizer_Type，跳过按优化器可视化。")


    # 3. 按学习率调度器 (Lr_Scheduler_Type) 对比平均准确率
    if 'Lr_Scheduler_Type' in df_valid.columns: # 注意：CSV中没有Lr_Scheduler_Type，这里假设可以从Combo_Name提取或添加
        # 尝试从 Combo_Name 提取 Lr_Scheduler_Type
        df_valid['Lr_Scheduler_Type'] = df_valid['Combo_Name'].apply(lambda x: x.split('__')[3] if len(x.split('__')) > 3 else 'Unknown')
        if not df_valid['Lr_Scheduler_Type'].empty and not all(df_valid['Lr_Scheduler_Type'] == 'Unknown'):
            plt.figure(figsize=(14, 6)) # 调度器类型可能较多，增加图表宽度
            sns.barplot(x='Lr_Scheduler_Type', y='Acceptance_Accuracy', data=df_valid)
            plt.title('平均验收准确率 (按学习率调度器)')
            plt.ylabel('准确率')
            plt.xlabel('学习率调度器类型')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right') # 旋转x轴标签以避免重叠
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'avg_accuracy_by_lr_schedule.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"图表已保存到 {plot_path}")
        else:
            print("警告: 无法从 Combo_Name 中提取有效的 Lr_Scheduler_Type，跳过按学习率调度器可视化。")


    # 4. 按学习率 (Learning_Rate) 和权重衰减 (Weight_Decay) 对比准确率 (散点图)
    # 确保 Learning_Rate 和 Weight_Decay 是数值类型
    # 注意：这些参数不在 acceptance_results.csv 的默认列中，但可以从 metadata.json 获取
    # 暂时跳过，如果需要，需在 acceptance.sh 中将这些参数添加到 CSV 或在此处通过其他方式获取
    print("提示: 学习率和权重衰减的可视化依赖于CSV中是否存在相关列，当前跳过。")
    # df_valid['Learning_Rate'] = pd.to_numeric(df_valid['Learning_Rate'], errors='coerce')
    # df_valid['Weight_Decay'] = pd.to_numeric(df_valid['Weight_Decay'], errors='coerce')
    # df_numeric_params = df_valid.dropna(subset=['Learning_Rate', 'Weight_Decay'])

    # if not df_numeric_params.empty:
    #     if len(df_numeric_params['Learning_Rate'].unique()) > 1:
    #         plt.figure(figsize=(10, 6))
    #         sns.scatterplot(x='Learning_Rate', y='Acceptance_Accuracy', hue='Loss_Type', style='Model_Type', data=df_numeric_params)
    #         plt.title('验收准确率 vs. 学习率')
    #         plt.ylabel('准确率')
    #         plt.xlabel('学习率')
    #         plt.ylim(0, 1.0)
    #         plt.xscale('log')
    #         plt.tight_layout()
    #         plot_path = os.path.join(output_dir, 'accuracy_vs_learning_rate.png')
    #         plt.savefig(plot_path)
    #         plt.close()
    #         print(f"图表已保存到 {plot_path}")

    #     if len(df_numeric_params['Weight_Decay'].unique()) > 1:
    #         plt.figure(figsize=(10, 6))
    #         sns.scatterplot(x='Weight_Decay', y='Acceptance_Accuracy', hue='Loss_Type', style='Model_Type', data=df_numeric_params)
    #         plt.title('验收准确率 vs. 权重衰减')
    #         plt.ylabel('准确率')
    #         plt.xlabel('权重衰减')
    #         plt.ylim(0, 1.0)
    #         plt.xscale('log')
    #         plt.tight_layout()
    #         plot_path = os.path.join(output_dir, 'accuracy_vs_weight_decay.png')
    #         plt.savefig(plot_path)
    #         plt.close()
    #         print(f"图表已保存到 {plot_path}")

    print(f"所有验收结果图表已生成并保存到目录: {output_dir}")

def visualize_single_image_comparison(csv_path: str, output_dir: str):
    """
    读取单张图片比较结果CSV，生成可视化图表。
    """
    if not os.path.exists(csv_path):
        print(f"错误：未找到单张图片比较结果文件 {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条单张图片比较结果。")
    except Exception as e:
        print(f"读取单张图片比较CSV文件失败: {e}")
        return

    # 确保分数是数值类型，处理可能的非数值条目 (如 N/A, ERROR)
    df['分数'] = pd.to_numeric(df['分数'], errors='coerce')
    df_valid = df.dropna(subset=['分数'])

    if df_valid.empty:
        print("没有有效的单张图片分数数据可供可视化。")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 绘制柱状图，展示每个模型的预测分数
    # 横轴：模型组合名称
    # 纵轴：分数
    # 每个柱子上标注预测标签
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x='模型组合名称', y='分数', hue='损失类型', data=df_valid, palette='viridis', dodge=True)
    
    plt.title(f'不同模型对单张图片的预测分数对比 (图片: {os.path.basename(csv_path).replace("single_image_comparison_results_", "").replace(".csv", "")})')
    plt.ylabel('预测分数 (相似度/置信度)')
    plt.xlabel('模型组合名称')
    plt.ylim(0, 1.0) # 分数通常在 0-1 之间
    plt.xticks(rotation=45, ha='right') # 旋转x轴标签以避免重叠

    # 在每个柱子上方添加预测标签
    for index, row in df_valid.iterrows():
        # 获取柱子的位置和高度
        # sns.barplot返回的ax.patches包含了每个bar的信息
        # 需要找到对应模型的bar来放置标签
        for bar in barplot.patches:
            # 检查bar的x位置是否与当前模型的x位置接近
            # 由于使用了hue='损失类型'和dodge=True，每个模型可能有多个bar
            # 需要更精确地匹配 bar 的模型组合名称和损失类型
            # 重新计算bar的中心x坐标
            model_name = barplot.get_xticklabels()[int(bar.get_x() + bar.get_width() / 2) % len(barplot.get_xticklabels())].get_text()
            if row['模型组合名称'] == model_name: # 简单的名称匹配，可能不精确，需要更健壮的匹配方式
                # 再次迭代，确保匹配到正确的hue组
                if barplot.get_legend() is not None:
                    hue_value = barplot.get_legend_handles_labels()[1][int(bar.get_x() + bar.get_width() / 2) // len(barplot.get_xticklabels())]
                    if row['损失类型'] == hue_value:
                        # 使用 .format() 替代 f-string，避免 linter 误报
                        text_to_display = "{}\n({:.2f})".format(row['预测标签'], row['分数'])
                        barplot.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                                     text_to_display, 
                                     color='black', ha='center', va='bottom', fontsize=9)
                else: # 没有hue的情况
                    # 使用 .format() 替代 f-string，避免 linter 误报
                    text_to_display = "{}\n({:.2f})".format(row['预测标签'], row['分数'])
                    barplot.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                                 text_to_display, 
                                 color='black', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'single_image_comparison_scores.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"单张图片比较图表已保存到 {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化模型评估结果脚本')
    parser.add_argument('--mode', type=str, default='acceptance', choices=['acceptance', 'single_image'],
                        help='''选择可视化模式: 'acceptance' (批量验收结果) 或 'single_image' (单张图片比较结果)。''')
    parser.add_argument('--csv_path', type=str, default='acceptance_results.csv',
                        help='''验收结果CSV文件的路径 (当 --mode 为 'acceptance' 时使用)。''')
    parser.add_argument('--single_image_csv_path', type=str, default=None,
                        help='''单张图片比较结果CSV文件的路径 (当 --mode 为 'single_image' 时使用)。''')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='图表保存目录。')
    cmd_args = parser.parse_args()

    if cmd_args.mode == 'acceptance':
        # 确保输出目录是 acceptance_plots
        acceptance_output_dir = os.path.join(cmd_args.output_dir, 'acceptance_plots')
        print(f"正在执行验收结果可视化，输出目录: {acceptance_output_dir}")
        visualize_acceptance_results(csv_path=cmd_args.csv_path, output_dir=acceptance_output_dir)
    elif cmd_args.mode == 'single_image':
        if cmd_args.single_image_csv_path is None:
            parser.error("在 'single_image' 模式下，必须通过 --single_image_csv_path 指定单张图片比较结果CSV文件。")
        # 确保输出目录是 single_image_comparison_plots
        single_image_output_dir = os.path.join(cmd_args.output_dir, 'single_image_comparison_plots')
        print(f"正在执行单张图片比较结果可视化，输出目录: {single_image_output_dir}")
        visualize_single_image_comparison(csv_path=cmd_args.single_image_csv_path, output_dir=single_image_output_dir)
    else:
        print("错误: 未知的可视化模式。")
