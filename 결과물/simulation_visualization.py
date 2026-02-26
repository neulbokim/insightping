"""
논문용 시각화 - STO 부동산 PF 리스크 분석
- 한글 지원
- 깔끔한 학술 스타일
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

# ===== 한글 폰트 설정 (강제) =====
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 학술 논문 스타일
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# ===== 데이터 로드 =====
file_path = './simulation_data.xlsx'
summary = pd.read_excel(file_path, sheet_name='Summary')
sto_benefit = pd.read_excel(file_path, sheet_name='STO_Benefit')
retail_risk = pd.read_excel(file_path, sheet_name='Retail_Risk')
distribution = pd.read_excel(file_path, sheet_name='Distribution')

print("✅ 데이터 로드 완료\n")

# ===== 색상 팔레트 (학술용) =====
COLORS = {
    'Perfect (100%)': '#2E7D32',  # 진한 녹색
    'Good (84%)': '#66BB6A',      # 연한 녹색
    'Recession (65%)': '#FFA726', # 주황색
    'Crisis (41%)': '#E53935',    # 빨간색
}

MARKET_ORDER = ['Perfect (100%)', 'Good (84%)', 'Recession (65%)', 'Crisis (41%)']

# ===== Figure 1: 금융기관 VaR95 비교 =====
print("📊 Figure 1: 금융기관 VaR95...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('금융기관 VaR95 (STO 비율별)', fontsize=14, fontweight='bold')

for idx, market in enumerate(MARKET_ORDER):
    ax = axes[idx // 2, idx % 2]
    
    market_data = summary[summary['Market'] == market].copy()
    
    # STO 비율을 숫자로 변환하여 정렬
    sto_numeric = []
    for ratio in market_data['STO_Ratio']:
        if ratio == 'Trad PF':
            sto_numeric.append(0)
        else:
            sto_numeric.append(int(ratio.replace('STO ', '').replace('%', '')))
    
    market_data['STO_Numeric'] = sto_numeric
    market_data = market_data.sort_values('STO_Numeric')
    
    # 라벨 생성
    sto_labels = [f'{x}%' for x in market_data['STO_Numeric']]
    
    x = np.arange(len(sto_labels))
    values = market_data['Financial_VaR95'].values / 10000  # 조 단위
    
    bars = ax.bar(x, values, color=COLORS[market], alpha=0.7, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('STO 비율', fontsize=10)
    ax.set_ylabel('VaR95 (조 원)', fontsize=10)
    ax.set_title(market, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sto_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure1_financial_var95.png', dpi=300, bbox_inches='tight')
print("  ✅ 저장: figure1_financial_var95.png\n")
plt.close()

# ===== Figure 2: 개인 투자자 VaR95 =====
print("📊 Figure 2: 개인 투자자 VaR95...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('개인 투자자 VaR95 (STO 비율별)', fontsize=14, fontweight='bold')

for idx, market in enumerate(MARKET_ORDER):
    ax = axes[idx // 2, idx % 2]
    
    market_data = summary[(summary['Market'] == market) & (summary['STO_Ratio'] != 'Trad PF')].copy()
    
    # STO 비율을 숫자로 변환하여 정렬
    sto_numeric = []
    for ratio in market_data['STO_Ratio']:
        sto_numeric.append(int(ratio.replace('STO ', '').replace('%', '')))
    
    market_data['STO_Numeric'] = sto_numeric
    market_data = market_data.sort_values('STO_Numeric')
    
    sto_labels = [f'{x}%' for x in market_data['STO_Numeric']]
    
    x = np.arange(len(sto_labels))
    values = market_data['Retail_VaR95_Absolute'].values
    
    bars = ax.bar(x, values, color=COLORS[market], alpha=0.7, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('STO 비율', fontsize=10)
    ax.set_ylabel('VaR95 (억 원)', fontsize=10)
    ax.set_title(market, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sto_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}',
                   ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure2_retail_var95.png', dpi=300, bbox_inches='tight')
print("  ✅ 저장: figure2_retail_var95.png\n")
plt.close()

# ===== Figure 3: 전체 시스템 리스크 비교 =====
print("📊 Figure 3: 전체 시스템 리스크 VaR95...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('전체 시스템 리스크 VaR95 (금융기관 + 개인)', fontsize=14, fontweight='bold')

for idx, market in enumerate(MARKET_ORDER):
    ax = axes[idx // 2, idx % 2]
    
    market_data = summary[summary['Market'] == market].copy()
    
    # STO 비율을 숫자로 변환하여 정렬
    sto_numeric = []
    for ratio in market_data['STO_Ratio']:
        if ratio == 'Trad PF':
            sto_numeric.append(0)
        else:
            sto_numeric.append(int(ratio.replace('STO ', '').replace('%', '')))
    
    market_data['STO_Numeric'] = sto_numeric
    market_data = market_data.sort_values('STO_Numeric')
    
    sto_labels = [f'{x}%' for x in market_data['STO_Numeric']]
    
    x = np.arange(len(sto_labels))
    values = market_data['Extended_Systemic_VaR95'].values / 10000  # 조 단위
    
    bars = ax.bar(x, values, color=COLORS[market], alpha=0.7, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('STO 비율', fontsize=10)
    ax.set_ylabel('VaR95 (조 원)', fontsize=10)
    ax.set_title(market, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sto_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure3_system_var95.png', dpi=300, bbox_inches='tight')
print("  ✅ 저장: figure3_system_var95.png\n")
plt.close()

# ===== Figure 4: STO Benefit (금융기관 VaR 감소율) =====
print("📊 Figure 4: STO Benefit (VaR 감소율)...")

fig, ax = plt.subplots(figsize=(10, 6))

# Heatmap 데이터 준비
pivot = sto_benefit.pivot(index='Market', columns='STO_Ratio', values='Reduction_Percent')
pivot = pivot.reindex(MARKET_ORDER)

# STO 비율을 숫자로 변환하여 정렬
sto_cols_sorted = sorted(pivot.columns, key=lambda x: int(x.replace('STO ', '').replace('%', '')))
pivot = pivot[sto_cols_sorted]

# STO 비율 라벨 정리
pivot.columns = [col.replace('STO ', '') for col in pivot.columns]

# Heatmap
im = ax.imshow(pivot.values, cmap='Greens', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_xticklabels(pivot.columns, fontsize=10)
ax.set_yticklabels(pivot.index, fontsize=10)

# 값 표시
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        if not pd.isna(val):
            text = ax.text(j, i, f'{val:.1f}%',
                          ha='center', va='center', 
                          color='black', fontsize=9, fontweight='bold')

ax.set_title('STO 도입 효과: 금융기관 VaR95 감소율 (%)', 
            fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('STO 비율', fontsize=11)
ax.set_ylabel('시장 상황', fontsize=11)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('감소율 (%)', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_sto_benefit.png', dpi=300, bbox_inches='tight')
print("  ✅ 저장: figure4_sto_benefit.png\n")
plt.close()

# ===== Figure 5: 개인 손실 확률 =====
print("📊 Figure 5: 개인 투자자 손실 확률...")

fig, ax = plt.subplots(figsize=(10, 6))

# Market별로 그룹화
for market in MARKET_ORDER:
    market_data = retail_risk[retail_risk['Market'] == market].copy()
    market_data = market_data.sort_values('STO_Ratio')
    
    sto_ratios = [int(ratio.replace('STO ', '').replace('%', '')) 
                  for ratio in market_data['STO_Ratio']]
    loss_probs = market_data['Loss_Probability'].values * 100  # %로 변환
    
    ax.plot(sto_ratios, loss_probs, 'o-', 
           label=market, color=COLORS[market], 
           linewidth=2, markersize=6)

ax.set_xlabel('STO 비율 (%)', fontsize=11)
ax.set_ylabel('손실 발생 확률 (%)', fontsize=11)
ax.set_title('개인 투자자 손실 확률 (전체 프로젝트 중)', 
            fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(retail_risk['Loss_Probability'].values * 100) * 1.1)

# 낮은 확률 강조
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(50, 11, '10% (참고선)', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('figure5_loss_probability.png', dpi=300, bbox_inches='tight')
print("  ✅ 저장: figure5_loss_probability.png\n")
plt.close()

# ===== Figure 6: 손실 분포 비교 (금융기관 vs 개인) - 1x2 =====
print("📊 Figure 6: 손실 분포 비교 (금융기관 vs 개인)...")

# Crisis, STO 30% 데이터
crisis_systemic = distribution[
    (distribution['Market'] == 'Crisis (41%)') & 
    (distribution['STO_Ratio'] == 'STO 30%') &
    (distribution['Type'] == 'systemic')
].copy()

crisis_retail = distribution[
    (distribution['Market'] == 'Crisis (41%)') & 
    (distribution['STO_Ratio'] == 'STO 30%') &
    (distribution['Type'] == 'retail')
].copy()

if len(crisis_systemic) > 0 and len(crisis_retail) > 0:
    crisis_systemic = crisis_systemic.sort_values('Percentile')
    crisis_retail = crisis_retail.sort_values('Percentile')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- 왼쪽: 금융기관 (Systemic) ---
    ax = axes[0]
    
    percentiles_sys = crisis_systemic['Percentile'].values
    values_sys = crisis_systemic['Value'].values / 10000  # 조 단위
    
    # 라인 플롯
    ax.plot(percentiles_sys, values_sys, 'bo-', linewidth=2, markersize=6, label='금융기관 손실')
    
    # 안전 구간 (0-90%)
    safe_idx = percentiles_sys <= 90
    if safe_idx.any():
        ax.fill_between(percentiles_sys[safe_idx], 0, values_sys[safe_idx],
                       alpha=0.2, color='blue', label='0-90% (안전)')
    
    # Tail 구간 (90-99%)
    tail_idx = percentiles_sys >= 90
    if tail_idx.any():
        ax.fill_between(percentiles_sys[tail_idx], 0, values_sys[tail_idx],
                       alpha=0.2, color='orange', label='90-99% (Tail Risk)')
    
    # 주요 percentile 표시
    for p in [0, 25, 50, 75, 90, 95, 99]:
        p_data = crisis_systemic[crisis_systemic['Percentile'] == p]
        if len(p_data) > 0:
            val = p_data['Value'].values[0] / 10000
            ax.axhline(y=val, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.text(2, val * 1.05, f'P{p}: {val:.2f}조', fontsize=8, color='gray')
    
    ax.set_xlabel('Percentile', fontsize=11)
    ax.set_ylabel('손실액 (조 원)', fontsize=11)
    ax.set_title('(a) 금융기관 손실 분포', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    
    # --- 오른쪽: 개인 투자자 (Retail) ---
    ax = axes[1]
    
    percentiles_ret = crisis_retail['Percentile'].values
    values_ret = crisis_retail['Value'].values
    
    # 라인 플롯
    ax.plot(percentiles_ret, values_ret, 'ro-', linewidth=2, markersize=6, label='개인 손실')
    
    # 안전 구간 (0-90%)
    safe_idx = percentiles_ret <= 90
    if safe_idx.any():
        ax.fill_between(percentiles_ret[safe_idx], 0, values_ret[safe_idx],
                       alpha=0.2, color='green', label='0-90% (안전)')
    
    # Tail 구간 (90-99%)
    tail_idx = percentiles_ret >= 90
    if tail_idx.any():
        ax.fill_between(percentiles_ret[tail_idx], 0, values_ret[tail_idx],
                       alpha=0.2, color='red', label='90-99% (Tail Risk)')
    
    # 주요 percentile 표시
    for p in [0, 25, 50, 75, 90, 95, 99]:
        p_data = crisis_retail[crisis_retail['Percentile'] == p]
        if len(p_data) > 0:
            val = p_data['Value'].values[0]
            ax.axhline(y=val, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.text(2, val * 1.05, f'P{p}: {val:.0f}억', fontsize=8, color='gray')
    
    ax.set_xlabel('Percentile', fontsize=11)
    ax.set_ylabel('손실액 (억 원)', fontsize=11)
    ax.set_title('(b) 개인 투자자 손실 분포', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    
    plt.suptitle('손실 분포 비교 (Crisis, STO 30%)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure6_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✅ 저장: figure6_distribution_comparison.png\n")
    plt.close()
else:
    print("  ⚠️  Crisis + STO 30% 데이터 없음, Figure 6 스킵\n")

# ===== Figure 7: 시스템 리스크 비교 (라인 그래프) =====
print("📊 Figure 7: 시스템 리스크 비교 (STO 비율별)...")

fig, ax = plt.subplots(figsize=(10, 6))

for market in MARKET_ORDER:
    market_data = summary[summary['Market'] == market].copy()
    
    # STO 비율 추출
    sto_values = []
    var_values = []
    
    for idx, row in market_data.iterrows():
        ratio = row['STO_Ratio']
        if ratio == 'Trad PF':
            sto_values.append(0)
        else:
            sto_values.append(int(ratio.replace('STO ', '').replace('%', '')))
        
        var_values.append(row['Extended_Systemic_VaR95'] / 10000)  # 조 단위
    
    # 정렬
    sorted_idx = np.argsort(sto_values)
    sto_values = np.array(sto_values)[sorted_idx]
    var_values = np.array(var_values)[sorted_idx]
    
    ax.plot(sto_values, var_values, 'o-', 
           label=market, color=COLORS[market],
           linewidth=2.5, markersize=6)

ax.set_xlabel('STO 비율 (%)', fontsize=11)
ax.set_ylabel('시스템 VaR95 (조 원)', fontsize=11)
ax.set_title('전체 시스템 리스크 추이 (STO 비율별)', 
            fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure7_system_trend.png', dpi=300, bbox_inches='tight')
print("  ✅ 저장: figure7_system_trend.png\n")
plt.close()

print("\n" + "="*60)
print("✅ 모든 시각화 완료!")
print("="*60)
print("\n생성된 파일:")
print("  1. figure1_financial_var95.png - 금융기관 VaR95")
print("  2. figure2_retail_var95.png - 개인 투자자 VaR95")
print("  3. figure3_system_var95.png - 전체 시스템 VaR95")
print("  4. figure4_sto_benefit.png - STO 도입 효과")
print("  5. figure5_loss_probability.png - 개인 손실 확률")
print("  6. figure6_retail_distribution.png - 개인 손실 분포 (Tail Risk)")
print("  7. figure7_system_trend.png - 시스템 리스크 추이")
print("\n논문용 고해상도(300 DPI) 이미지입니다.")