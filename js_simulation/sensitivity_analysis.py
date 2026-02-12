"""
민감도 분석 (Sensitivity Analysis)
STO 도입 시 리스크에 영향을 미치는 주요 변수 파악

방법론:
1. One-at-a-Time (OAT) 민감도 분석
2. Tornado 다이어그램 (변수별 영향도)
3. 2D 민감도 맵 (2개 변수 동시 변화)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pf_simulation_v2 import SimulationParams, ImprovedPFSimulation
from pf_analysis_v2 import ImprovedRiskAnalyzer
import time

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_list = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in font_list:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'NanumGothic' in font_list:
    plt.rcParams['font.family'] = 'NanumGothic'
elif 'AppleGothic' in font_list:
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ===== 기준 시나리오 (Baseline) =====
BASELINE_PARAMS = {
    'n_simulations': 5000,
    'n_projects': 100,
    'T': 16,
    'sto_ratio': 0.28,  # STO 28%
    'use_logistic_sales': False,
    'mu_sales_base': 0.00,
    'sigma_sales': 0.25,
    'recovery_rate_base': 0.25,
    'collateral_ratio': 0.30,
    'rho_base': 0.30,
    'fire_sale_base': 0.50,
}


# ===== 민감도 분석 변수 범위 =====
SENSITIVITY_VARIABLES = {
    'sto_ratio': {
        'label': 'STO 비율 (%)',
        'baseline': 0.28,
        'range': [0.10, 0.15, 0.20, 0.28, 0.35, 0.40, 0.45, 0.50],
        'unit': '%',
        'format': lambda x: f'{x*100:.0f}%'
    },
    'mu_sales_base': {
        'label': '분양률 성장률 (%/분기)',
        'baseline': 0.00,
        'range': [-0.05, -0.03, -0.01, 0.00, 0.01, 0.03, 0.05],
        'unit': '%',
        'format': lambda x: f'{x*100:+.1f}%'
    },
    'sigma_sales': {
        'label': '분양률 변동성',
        'baseline': 0.25,
        'range': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'unit': '',
        'format': lambda x: f'{x:.2f}'
    },
    'recovery_rate_base': {
        'label': '기본 회수율 (%)',
        'baseline': 0.25,
        'range': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'unit': '%',
        'format': lambda x: f'{x*100:.0f}%'
    },
    'collateral_ratio': {
        'label': '담보 비율 (%)',
        'baseline': 0.30,
        'range': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        'unit': '%',
        'format': lambda x: f'{x*100:.0f}%'
    },
    'rho_base': {
        'label': '기본 상관계수',
        'baseline': 0.30,
        'range': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        'unit': '',
        'format': lambda x: f'{x:.2f}'
    },
    'fire_sale_base': {
        'label': '화급매각 할인율 (%)',
        'baseline': 0.50,
        'range': [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        'unit': '%',
        'format': lambda x: f'{x*100:.0f}%'
    },
}


def run_single_scenario(param_overrides):
    """단일 시나리오 실행"""
    params_dict = BASELINE_PARAMS.copy()
    params_dict.update(param_overrides)
    
    params = SimulationParams(**params_dict)
    sim = ImprovedPFSimulation(params, use_sto=True)
    results = sim.run_simulation()
    
    analyzer = ImprovedRiskAnalyzer(results, params)
    metrics = analyzer.calculate_all_metrics()
    
    # 확장 시스템 VaR 계산
    financial_var = metrics['VaR_95']
    retail_var = metrics['retail_VaR_95']
    extended_var = financial_var + retail_var
    system_change = retail_var
    
    return {
        'Financial_VaR95': financial_var,
        'Retail_VaR95': retail_var,
        'Extended_VaR95': extended_var,
        'System_Risk_Change': system_change,
        'Retail_Loss_Rate_VaR95': metrics['retail_loss_rate_VaR95'],
        'Retail_Loss_Rate_ES95': metrics['retail_loss_rate_ES95'],
    }


def run_oat_sensitivity_analysis():
    """One-at-a-Time 민감도 분석"""
    print("="*80)
    print("One-at-a-Time (OAT) 민감도 분석 시작")
    print("="*80)
    
    results = {}
    
    # 기준 시나리오 실행
    print("\n[Baseline] 기준 시나리오 실행 중...")
    baseline_results = run_single_scenario({})
    
    print(f"  Financial VaR95: {baseline_results['Financial_VaR95']:,.0f}억")
    print(f"  Retail VaR95: {baseline_results['Retail_VaR95']:,.0f}억")
    print(f"  Extended VaR95: {baseline_results['Extended_VaR95']:,.0f}억")
    
    results['baseline'] = baseline_results
    
    # 각 변수별 민감도 분석
    for var_name, var_info in SENSITIVITY_VARIABLES.items():
        print(f"\n[{var_info['label']}] 민감도 분석 중...")
        
        var_results = []
        
        for value in var_info['range']:
            print(f"  {var_info['format'](value)} ", end='', flush=True)
            
            scenario_results = run_single_scenario({var_name: value})
            scenario_results['value'] = value
            var_results.append(scenario_results)
            
            print("✓", end='', flush=True)
        
        print()  # 줄바꿈
        results[var_name] = var_results
    
    print("\n" + "="*80)
    print("✅ OAT 민감도 분석 완료")
    print("="*80)
    
    return results, baseline_results


def calculate_sensitivity_metrics(results, baseline):
    """민감도 지표 계산"""
    sensitivity_metrics = []
    
    for var_name, var_results in results.items():
        if var_name == 'baseline':
            continue
        
        var_info = SENSITIVITY_VARIABLES[var_name]
        
        # 각 출력 지표별 민감도
        for output_metric in ['Financial_VaR95', 'Retail_VaR95', 'Extended_VaR95', 'System_Risk_Change']:
            values = [r[output_metric] for r in var_results]
            
            # 민감도 = (Max - Min) / Baseline
            sensitivity = (max(values) - min(values)) / baseline[output_metric] * 100
            
            # 기울기 (선형 근사)
            x = var_info['range']
            y = values
            slope = np.polyfit(x, y, 1)[0]
            
            sensitivity_metrics.append({
                'Variable': var_info['label'],
                'Output_Metric': output_metric,
                'Sensitivity_Pct': sensitivity,
                'Slope': slope,
                'Min_Value': min(values),
                'Max_Value': max(values),
                'Range': max(values) - min(values),
            })
    
    df = pd.DataFrame(sensitivity_metrics)
    return df


def create_tornado_diagram(sensitivity_df, baseline):
    """Tornado 다이어그램 생성"""
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.suptitle('민감도 분석: Tornado 다이어그램 (STO 28% 기준)\n변수별 영향력 순위', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    output_metrics = ['Financial_VaR95', 'Retail_VaR95', 'Extended_VaR95', 'System_Risk_Change']
    metric_labels = ['금융기관 VaR95', '개인 투자자 VaR95', '확장 시스템 VaR95', '시스템 리스크 증가']
    
    for idx, (metric, label) in enumerate(zip(output_metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # 해당 지표에 대한 민감도 필터링
        metric_data = sensitivity_df[sensitivity_df['Output_Metric'] == metric].copy()
        metric_data = metric_data.sort_values('Sensitivity_Pct', ascending=True)
        
        # Tornado 차트
        y_pos = np.arange(len(metric_data))
        
        # 최소값과 최대값의 변화량
        baseline_value = baseline[metric]
        low_change = (metric_data['Min_Value'] - baseline_value) / baseline_value * 100
        high_change = (metric_data['Max_Value'] - baseline_value) / baseline_value * 100
        
        # 막대 그래프
        bars_high = ax.barh(y_pos, high_change, left=0, color='lightcoral', 
                           edgecolor='black', linewidth=1.5, label='증가 (Max)', alpha=0.8)
        bars_low = ax.barh(y_pos, low_change, left=0, color='lightblue', 
                          edgecolor='black', linewidth=1.5, label='감소 (Min)', alpha=0.8)
        
        # 기준선
        ax.axvline(0, color='black', linewidth=2.5, linestyle='--', alpha=0.8)
        
        # 레이블 (순위 추가)
        labels_with_rank = [f'{i+1}. {var}' for i, var in enumerate(metric_data['Variable'])]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_with_rank, fontsize=11)
        ax.set_xlabel('기준 대비 변화율 (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{label}\n(기준: {baseline_value:,.0f}억)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 값 표시 (더 큰 폰트)
        for i, (low, high) in enumerate(zip(low_change, high_change)):
            if abs(low) > 3:  # 3% 이상만 표시
                ax.text(low - 2, i, f'{low:.1f}%', 
                       ha='right', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            if abs(high) > 3:
                ax.text(high + 2, i, f'{high:.1f}%', 
                       ha='left', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 범위 표시
        ax.set_xlim(min(low_change.min(), high_change.min()) * 1.2,
                   max(low_change.max(), high_change.max()) * 1.2)
    
    plt.tight_layout()
    plt.savefig('sensitivity_tornado.png', dpi=150, bbox_inches='tight')
    print("\n✅ Tornado 다이어그램 저장: sensitivity_tornado.png")


def create_sensitivity_curves(results, baseline):
    """민감도 곡선 그래프"""
    
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    output_metrics = ['Financial_VaR95', 'Retail_VaR95', 'Extended_VaR95', 'System_Risk_Change']
    metric_labels = ['금융기관 VaR95 (억원)', '개인 투자자 VaR95 (억원)', 
                     '확장 시스템 VaR95 (억원)', '시스템 리스크 증가 (억원)']
    
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    for row, (metric, label) in enumerate(zip(output_metrics, metric_labels)):
        
        for col, (var_name, var_info) in enumerate(SENSITIVITY_VARIABLES.items()):
            if col >= 4:  # 4열까지만
                continue
            
            ax = fig.add_subplot(gs[row, col])
            
            var_results = results[var_name]
            x_values = [r['value'] for r in var_results]
            y_values = [r[metric] for r in var_results]
            
            # 기준선
            baseline_x = var_info['baseline']
            baseline_y = baseline[metric]
            
            # 곡선
            ax.plot(x_values, y_values, 'o-', linewidth=2.5, 
                   markersize=6, color=colors[col], label=var_info['label'])
            
            # 기준점 강조
            ax.plot(baseline_x, baseline_y, 'r*', markersize=15, 
                   label='Baseline', zorder=5)
            
            # 스타일
            ax.set_xlabel(var_info['label'], fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            
            # X축 포맷
            ax.set_xticks(x_values[::2])  # 격자로 표시
            ax.set_xticklabels([var_info['format'](v) for v in x_values[::2]], 
                              rotation=45, fontsize=9)
    
    # 나머지 3개 변수 (두 번째 행)
    remaining_vars = list(SENSITIVITY_VARIABLES.items())[4:]
    for col, (var_name, var_info) in enumerate(remaining_vars):
        
        for row, (metric, label) in enumerate(zip(output_metrics, metric_labels)):
            
            ax = fig.add_subplot(gs[row, col])
            
            var_results = results[var_name]
            x_values = [r['value'] for r in var_results]
            y_values = [r[metric] for r in var_results]
            
            baseline_x = var_info['baseline']
            baseline_y = baseline[metric]
            
            ax.plot(x_values, y_values, 'o-', linewidth=2.5, 
                   markersize=6, color=colors[col+4], label=var_info['label'])
            ax.plot(baseline_x, baseline_y, 'r*', markersize=15, 
                   label='Baseline', zorder=5)
            
            ax.set_xlabel(var_info['label'], fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            
            ax.set_xticks(x_values[::2])
            ax.set_xticklabels([var_info['format'](v) for v in x_values[::2]], 
                              rotation=45, fontsize=9)
    
    fig.suptitle('민감도 분석: 변수별 영향 곡선 (STO 28% 기준)', 
                 fontsize=18, fontweight='bold', y=0.998)
    
    plt.savefig('sensitivity_curves.png', dpi=150, bbox_inches='tight')
    print("✅ 민감도 곡선 저장: sensitivity_curves.png")


def export_sensitivity_to_excel(sensitivity_df, results, baseline):
    """Excel로 결과 내보내기"""
    
    with pd.ExcelWriter('sensitivity_analysis_results.xlsx', engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        summary = sensitivity_df.pivot_table(
            index='Variable', 
            columns='Output_Metric', 
            values='Sensitivity_Pct'
        )
        summary.to_excel(writer, sheet_name='Sensitivity_Summary')
        
        # Sheet 2: Detailed Results
        sensitivity_df.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
        
        # Sheet 3-9: 각 변수별 상세 결과
        for var_name, var_results in results.items():
            if var_name == 'baseline':
                continue
            
            var_df = pd.DataFrame(var_results)
            var_df.to_excel(writer, sheet_name=var_name[:31], index=False)
    
    print("✅ Excel 파일 저장: sensitivity_analysis_results.xlsx")


def print_sensitivity_insights(sensitivity_df):
    """핵심 인사이트 출력"""
    
    print("\n" + "="*80)
    print("📊 민감도 분석 핵심 인사이트 (Tornado Diagram 기반)")
    print("="*80)
    
    for metric, label in [('Financial_VaR95', '금융기관 VaR95'),
                          ('Retail_VaR95', '개인 투자자 VaR95'),
                          ('Extended_VaR95', '확장 시스템 VaR95'),
                          ('System_Risk_Change', '시스템 리스크 증가')]:
        
        metric_data = sensitivity_df[sensitivity_df['Output_Metric'] == metric]
        top3 = metric_data.nlargest(3, 'Sensitivity_Pct')
        
        print(f"\n{'='*80}")
        print(f"🎯 {label}에 가장 큰 영향을 미치는 변수 (Top 3):")
        print(f"{'='*80}")
        
        for i, row in enumerate(top3.itertuples(), 1):
            rank_emoji = ['🥇', '🥈', '🥉'][i-1]
            print(f"\n{rank_emoji} {i}위: {row.Variable}")
            print(f"   민감도: ±{row.Sensitivity_Pct:.1f}% (영향력 지수)")
            print(f"   변동 범위: {row.Min_Value:,.0f}억 ~ {row.Max_Value:,.0f}억")
            print(f"   변화폭: {row.Range:,.0f}억")
            
            # 해석
            if row.Sensitivity_Pct > 50:
                interpretation = "⚠️  매우 높은 영향 - 필수 모니터링 대상"
            elif row.Sensitivity_Pct > 30:
                interpretation = "⚠️  높은 영향 - 중요 관리 변수"
            elif row.Sensitivity_Pct > 15:
                interpretation = "✓ 중간 영향 - 주의 필요"
            else:
                interpretation = "✓ 낮은 영향"
            
            print(f"   {interpretation}")
    
    # 전체 요약
    print(f"\n{'='*80}")
    print("📌 종합 분석 결과:")
    print(f"{'='*80}")
    
    # 모든 지표에 공통적으로 영향력 큰 변수
    all_top_vars = []
    for metric in ['Financial_VaR95', 'Retail_VaR95', 'Extended_VaR95', 'System_Risk_Change']:
        metric_data = sensitivity_df[sensitivity_df['Output_Metric'] == metric]
        top3 = metric_data.nlargest(3, 'Sensitivity_Pct')
        all_top_vars.extend(top3['Variable'].tolist())
    
    from collections import Counter
    var_counts = Counter(all_top_vars)
    most_common = var_counts.most_common(3)
    
    print("\n✅ 전체 리스크 지표에 공통적으로 영향력이 큰 변수:")
    for i, (var, count) in enumerate(most_common, 1):
        print(f"   {i}. {var} (4개 지표 중 {count}개에서 Top 3)")
    
    print("\n💡 리스크 관리 권고사항:")
    print("   1. 상위 3개 변수를 집중 모니터링")
    print("   2. 하위 변수는 기준값 유지로 충분")
    print("   3. 토네이도 차트에서 비대칭성 확인 → 방향성 있는 정책 수립")


def main():
    """메인 실행"""
    
    start_time = time.time()
    
    print("="*80)
    print("민감도 분석 (Sensitivity Analysis)")
    print("STO 도입 시 리스크 영향 변수 파악")
    print("="*80)
    
    # 1. OAT 민감도 분석 실행
    results, baseline = run_oat_sensitivity_analysis()
    
    # 2. 민감도 지표 계산
    sensitivity_df = calculate_sensitivity_metrics(results, baseline)
    
    # 3. Tornado 다이어그램
    create_tornado_diagram(sensitivity_df, baseline)
    
    # 4. 민감도 곡선
    create_sensitivity_curves(results, baseline)
    
    # 5. Excel 내보내기
    export_sensitivity_to_excel(sensitivity_df, results, baseline)
    
    # 6. 인사이트 출력
    print_sensitivity_insights(sensitivity_df)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  총 소요 시간: {elapsed/60:.1f}분")
    
    print("\n" + "="*80)
    print("✅ 민감도 분석 완료!")
    print("="*80)


if __name__ == "__main__":
    main()