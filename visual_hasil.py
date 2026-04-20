import json
import matplotlib.pyplot as plt
import pandas as pd

def load_metrics(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

df2 = load_metrics('/home/marh/Project/Vim-TA-MARH/det/work_dirs/testi_2/metrics2.json')
df3 = load_metrics('/home/marh/Project/Vim-TA-MARH/det/work_dirs/testi_2/metrics3.json')
df4 = load_metrics('/home/marh/Project/Vim-TA-MARH/det/work_dirs/testi_4/metrics.json')

train2 = df2[df2['total_loss'].notna()]
eval2 = df2[df2['bbox/AP'].notna()]

train3 = df3[df3['total_loss'].notna()]
eval3 = df3[df3['bbox/AP'].notna()]

train4 = df4[df4['total_loss'].notna()]
eval4 = df4[df4['bbox/AP'].notna()]

# Aggregate cascade losses
cls_cols2 = [c for c in train2.columns if c.startswith('loss_cls_stage')]
box_cols2 = [c for c in train2.columns if c.startswith('loss_box_reg_stage')]
cls_cols3 = [c for c in train3.columns if c.startswith('loss_cls_stage')]
box_cols3 = [c for c in train3.columns if c.startswith('loss_box_reg_stage')]
cls_cols4 = [c for c in train4.columns if c.startswith('loss_cls_stage')]
box_cols4 = [c for c in train4.columns if c.startswith('loss_box_reg_stage')]

train2['loss_cls_sum'] = train2[cls_cols2].sum(axis=1)
train2['loss_box_sum'] = train2[box_cols2].sum(axis=1)
train3['loss_cls_sum'] = train3[cls_cols3].sum(axis=1)
train3['loss_box_sum'] = train3[box_cols3].sum(axis=1)
train4['loss_cls_sum'] = train4[cls_cols4].sum(axis=1)
train4['loss_box_sum'] = train4[box_cols4].sum(axis=1)

# Plot 1: Total Loss
plt.figure(figsize=(10, 6))
plt.plot(train2['iteration'], train2['total_loss'], label='Experiment 2 (metrics2)', alpha=0.6)
plt.plot(train3['iteration'], train3['total_loss'], label='Experiment 3 (metrics3)', alpha=0.6)
plt.plot(train4['iteration'], train4['total_loss'], label='Experiment 4 (metrics4)', alpha=0.6)
plt.title('Total Loss Comparison')
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('total_loss_comparison.png')
plt.close()

# Plot 2: mAP (bbox/AP)
plt.figure(figsize=(10, 6))
plt.plot(eval2['iteration'], eval2['bbox/AP'], marker='o', label='Experiment 2 (metrics2)')
plt.plot(eval3['iteration'], eval3['bbox/AP'], marker='s', label='Experiment 3 (metrics3)')
plt.plot(eval4['iteration'], eval4['bbox/AP'], marker='^', label='Experiment 4 (metrics4)')
plt.title('mAP (bbox/AP) Comparison')
plt.xlabel('Iteration')
plt.ylabel('mAP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('map_comparison.png')
plt.close()

# Plot 3: AP50
plt.figure(figsize=(10, 6))
plt.plot(eval2['iteration'], eval2['bbox/AP50'], marker='o', label='Experiment 2 (metrics2)')
plt.plot(eval3['iteration'], eval3['bbox/AP50'], marker='s', label='Experiment 3 (metrics3)')
plt.plot(eval4['iteration'], eval4['bbox/AP50'], marker='^', label='Experiment 4 (metrics4)')
plt.title('AP50 Comparison')
plt.xlabel('Iteration')
plt.ylabel('AP50')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ap50_comparison.png')
plt.close()

# Plot 4: AP75
plt.figure(figsize=(10, 6))
plt.plot(eval2['iteration'], eval2['bbox/AP75'], marker='o', label='Experiment 2 (metrics2)')
plt.plot(eval3['iteration'], eval3['bbox/AP75'], marker='s', label='Experiment 3 (metrics3)')
plt.plot(eval4['iteration'], eval4['bbox/AP75'], marker='^', label='Experiment 4 (metrics4)')
plt.title('AP75 Comparison')
plt.xlabel('Iteration')
plt.ylabel('AP75')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ap75_comparison.png')
plt.close()

# Plot 5: Classification Loss (sum of stage0/1/2)
plt.figure(figsize=(10, 6))
plt.plot(train2['iteration'], train2['loss_cls_sum'], label='Experiment 2 (metrics2)', alpha=0.7)
plt.plot(train3['iteration'], train3['loss_cls_sum'], label='Experiment 3 (metrics3)', alpha=0.7)
plt.plot(train4['iteration'], train4['loss_cls_sum'], label='Experiment 4 (metrics4)', alpha=0.7)
plt.title('Classification Loss Comparison (Sum of Stages)')
plt.xlabel('Iteration')
plt.ylabel('Loss CLS (sum)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cls_loss_comparison.png')
plt.close()

# Plot 6: Box Regression Loss (sum of stage0/1/2)
plt.figure(figsize=(10, 6))
plt.plot(train2['iteration'], train2['loss_box_sum'], label='Experiment 2 (metrics2)', alpha=0.7)
plt.plot(train3['iteration'], train3['loss_box_sum'], label='Experiment 3 (metrics3)', alpha=0.7)
plt.plot(train4['iteration'], train4['loss_box_sum'], label='Experiment 4 (metrics4)', alpha=0.7)
plt.title('Box Regression Loss Comparison (Sum of Stages)')
plt.xlabel('Iteration')
plt.ylabel('Loss BOX (sum)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('box_loss_comparison.png')
plt.close()

# Plot 7: Overall (all metrics in one figure)
fig, axes = plt.subplots(2, 3, figsize=(22, 11))

ax = axes[0, 0]
ax.plot(train2['iteration'], train2['total_loss'], label='Experiment 2 (metrics2)', alpha=0.7)
ax.plot(train3['iteration'], train3['total_loss'], label='Experiment 3 (metrics3)', alpha=0.7)
ax.plot(train4['iteration'], train4['total_loss'], label='Experiment 4 (metrics4)', alpha=0.7)
ax.set_title('Total Loss Comparison')
ax.set_xlabel('Iteration')
ax.set_ylabel('Total Loss')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(eval2['iteration'], eval2['bbox/AP'], marker='o', label='Experiment 2 (metrics2)')
ax.plot(eval3['iteration'], eval3['bbox/AP'], marker='s', label='Experiment 3 (metrics3)')
ax.plot(eval4['iteration'], eval4['bbox/AP'], marker='^', label='Experiment 4 (metrics4)')
ax.set_title('mAP (bbox/AP) Comparison')
ax.set_xlabel('Iteration')
ax.set_ylabel('mAP')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.plot(eval2['iteration'], eval2['bbox/AP50'], marker='o', label='Experiment 2 (metrics2)')
ax.plot(eval3['iteration'], eval3['bbox/AP50'], marker='s', label='Experiment 3 (metrics3)')
ax.plot(eval4['iteration'], eval4['bbox/AP50'], marker='^', label='Experiment 4 (metrics4)')
ax.set_title('AP50 Comparison')
ax.set_xlabel('Iteration')
ax.set_ylabel('AP50')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(eval2['iteration'], eval2['bbox/AP75'], marker='o', label='Experiment 2 (metrics2)')
ax.plot(eval3['iteration'], eval3['bbox/AP75'], marker='s', label='Experiment 3 (metrics3)')
ax.plot(eval4['iteration'], eval4['bbox/AP75'], marker='^', label='Experiment 4 (metrics4)')
ax.set_title('AP75 Comparison')
ax.set_xlabel('Iteration')
ax.set_ylabel('AP75')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(train2['iteration'], train2['loss_cls_sum'], label='Experiment 2 (metrics2)', alpha=0.7)
ax.plot(train3['iteration'], train3['loss_cls_sum'], label='Experiment 3 (metrics3)', alpha=0.7)
ax.plot(train4['iteration'], train4['loss_cls_sum'], label='Experiment 4 (metrics4)', alpha=0.7)
ax.set_title('Classification Loss Comparison (Sum of Stages)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss CLS (sum)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1, 2]
ax.plot(train2['iteration'], train2['loss_box_sum'], label='Experiment 2 (metrics2)', alpha=0.7)
ax.plot(train3['iteration'], train3['loss_box_sum'], label='Experiment 3 (metrics3)', alpha=0.7)
ax.plot(train4['iteration'], train4['loss_box_sum'], label='Experiment 4 (metrics4)', alpha=0.7)
ax.set_title('Box Regression Loss Comparison (Sum of Stages)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss BOX (sum)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle('Overall Comparison: Experiment 2 vs 3 vs 4', fontsize=16)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig('overall_comparison.svg', format='svg')
plt.close(fig)

final_metrics = {
    'metrics2': {
        'final_loss': train2['total_loss'].iloc[-1],
        'max_mAP': eval2['bbox/AP'].max(),
        'final_mAP': eval2['bbox/AP'].iloc[-1],
        'max_AP50': eval2['bbox/AP50'].max(),
        'final_AP50': eval2['bbox/AP50'].iloc[-1],
        'max_AP75': eval2['bbox/AP75'].max(),
        'final_AP75': eval2['bbox/AP75'].iloc[-1],
        'final_loss_cls_sum': train2['loss_cls_sum'].iloc[-1],
        'final_loss_box_sum': train2['loss_box_sum'].iloc[-1],
    },
    'metrics3': {
        'final_loss': train3['total_loss'].iloc[-1],
        'max_mAP': eval3['bbox/AP'].max(),
        'final_mAP': eval3['bbox/AP'].iloc[-1],
        'max_AP50': eval3['bbox/AP50'].max(),
        'final_AP50': eval3['bbox/AP50'].iloc[-1],
        'max_AP75': eval3['bbox/AP75'].max(),
        'final_AP75': eval3['bbox/AP75'].iloc[-1],
        'final_loss_cls_sum': train3['loss_cls_sum'].iloc[-1],
        'final_loss_box_sum': train3['loss_box_sum'].iloc[-1],
    },
    'metrics4': {
        'final_loss': train4['total_loss'].iloc[-1],
        'max_mAP': eval4['bbox/AP'].max(),
        'final_mAP': eval4['bbox/AP'].iloc[-1],
        'max_AP50': eval4['bbox/AP50'].max(),
        'final_AP50': eval4['bbox/AP50'].iloc[-1],
        'max_AP75': eval4['bbox/AP75'].max(),
        'final_AP75': eval4['bbox/AP75'].iloc[-1],
        'final_loss_cls_sum': train4['loss_cls_sum'].iloc[-1],
        'final_loss_box_sum': train4['loss_box_sum'].iloc[-1],
    }
}
print(json.dumps(final_metrics, indent=2))