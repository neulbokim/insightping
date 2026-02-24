"""
민감도 분석 (Sensitivity Analysis) - Tail Risk 포함
STO 도입 시 리스크에 영향을 미치는 주요 변수 파악

개선사항:
- Retail VaR99, Max, Tail 배율 추가
- Crisis 시나리오 기본 (한 번에 실행)
- Tornado + 곡선 + Excel 통합
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pf_simulation_v2 import SimulationParams, ImprovedPFSimulation
from pf_analysis_v2 import ImprovedRiskAnalyzer
import time

# 한글 폰트
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


# ===== 출력 지표 (Tail Risk 포함) =====
OUTPUT_METRICS = [
    ('Financial_VaR95',    '금융기관 VaR95'),
    ('Retail_VaR95',       '개인 VaR95'),
    ('Retail_VaR99',       '개인 VaR99'),
    ('Retail_Max',         '개인 Max'),
    ('Retail_Tail_Mult',   '개인 Tail배율'),
    ('Extended_VaR95',     '확장 VaR95'),
]


# ===== 기준 시나리오: Crisis =====
BASELINE_PARAMS = {
    'n_simulations': 5000,
    'n_projects': 100,
    'T': 16,
    'sto_ratio': 0.30,
    'use_logistic_sales': False,
    'mu_sales_base': 0.00,
    'sigma_sales': 0.25,
    'initial_sales': 0.15,
    'recovery_rate_base': 0.25,
    'collateral_ratio': 0.30,
    'rho_base': 0.30,
    'fire_sale_base': 0.50,
}


# ===== 민감도 변수 =====
SENSITIVITY_VARIABLES = {
    'sto_ratio': {
        'label': 'STO 비율',
        'baseline': 0.28,
        'range': [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.9, 1],
        'format': lambda x: f'{x*100:.0f}%'
    },
    'mu_sales_base': {
        'label': '분양률 성장률',
        'baseline': 0.00,
        'range': [-0.05, -0.03, -0.01, 0.00, 0.01, 0.03, 0.05],
        'format': lambda x: f'{x*100:+.1f}%'
    },
    'sigma_sales': {
        'label': '분양률 변동성',
        'baseline': 0.25,
        'range': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'format': lambda x: f'{x:.2f}'
    },
    'recovery_rate_base': {
        'label': '기본 회수율',
        'baseline': 0.25,
        'range': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'format': lambda x: f'{x*100:.0f}%'
    },
    'collateral_ratio': {
        'label': '담보 비율',
        'baseline': 0.30,
        'range': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        'format': lambda x: f'{x*100:.0f}%'
    },
    'rho_base': {
        'label': '기본 상관계수',
        'baseline': 0.30,
        'range': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        'format': lambda x: f'{x:.2f}'
    },
    'fire_sale_base': {
        'label': '급매각 할인율',
        'baseline': 0.50,
        'range': [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        'format': lambda x: f'{x*100:.0f}%'
    },
}


def run_scenario(baseline_params, overrides=None):
    """단일 시나리오 실행 (Tail Risk 포함)"""
    params_dict = baseline_params.copy()
    if overrides:
        params_dict.update(overrides)
    
    params = SimulationParams(**params_dict)
    use_sto = params.sto_ratio > 0
    sim = ImprovedPFSimulation(params, use_sto=use_sto)
    results = sim.run_simulation()
    
    analyzer = ImprovedRiskAnalyzer(results, params)
    metrics = analyzer.calculate_all_metrics()
    
    # 금융
    fin_var95 = metrics['VaR_95']
    
    # 개인 (Tail Risk 포함)
    if use_sto:
        retail_loss = results['losses']['retail_loss']
        retail_total = retail_loss[:, :, -1].sum(axis=1)
        
        ret_var95 = np.percentile(retail_total, 95)
        ret_var99 = np.percentile(retail_total, 99)
        ret_max = retail_total.max()
        ret_tail_mult = ret_max / ret_var95 if ret_var95 > 0 else 0
    else:
        ret_var95 = 0
        ret_var99 = 0
        ret_max = 0
        ret_tail_mult = 0
    
    del results
    
    return {
        'Financial_VaR95': fin_var95,
        'Retail_VaR95': ret_var95,
        'Retail_VaR99': ret_var99,
        'Retail_Max': ret_max,
        'Retail_Tail_Mult': ret_tail_mult,
        'Extended_VaR95': fin_var95 + ret_var95,
    }


def run_oat_analysis(baseline_params, variables):
    """One-at-a-Time 민감도 분석"""
    print("="*80)
    print("민감도 분석 (Crisis 시나리오)")
    print("="*80)
    
    # Baseline
    print("\n[Baseline] 실행...")
    baseline = run_scenario(baseline_params)
    print(f"  Financial VaR95: {baseline['Financial_VaR95']:,.0f}억")
    print(f"  Retail VaR95:    {baseline['Retail_VaR95']:,.0f}억")
    print(f"  Retail VaR99:    {baseline['Retail_VaR99']:,.0f}억")
    print(f"  Retail Max:      {baseline['Retail_Max']:,.0f}억")
    print(f"  Retail Tail:     {baseline['Retail_Tail_Mult']:.2f}x")
    
    # 변수별 분석
    results = {}
    total = sum(len(v['range']) for v in variables.values())
    done = 0
    
    for var_name, var_info in variables.items():
        print(f"\n[{var_info['label']}]", flush=True)
        var_results = []
        
        for value in var_info['range']:
            tag = var_info['format'](value)
            res = run_scenario(baseline_params, {var_name: value})
            res['value'] = value
            var_results.append(res)
            
            done += 1
            print(f"  {tag:>8} → F:{res['Financial_VaR95']:>7,.0f} "
                  f"R:{res['Retail_VaR95']:>7,.0f} "
                  f"Tail:{res['Retail_Tail_Mult']:>5.1f}x "
                  f"[{done}/{total}]")
        
        results[var_name] = var_results
    
    print("\n" + "="*80)
    print("✅ 민감도 분석 완료")
    print("="*80)
    
    return results, baseline


def calc_sensitivity_df(results, baseline, variables):
    """민감도 지표 계산"""
    rows = []
    
    for var_name, var_results in results.items():
        var_info = variables[var_name]
        
        for metric_key, metric_label in OUTPUT_METRICS:
            values = [r[metric_key] for r in var_results]
            base_val = baseline[metric_key]
            
            rng = max(values) - min(values)
            pct = (rng / abs(base_val) * 100) if base_val != 0 else 0.0
            
            rows.append({
                'Variable': var_info['label'],
                'Metric': metric_key,
                'Metric_Label': metric_label,
                'Sensitivity_Pct': pct,
                'Min': min(values),
                'Max': max(values),
                'Range': rng,
                'Baseline': base_val,
            })
    
    return pd.DataFrame(rows)


def plot_tornado(sens_df, baseline):
    """Tornado 다이어그램 (6개 지표)"""
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle('Tornado 다이어그램 - Crisis 시나리오, STO 28%',
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes_flat = axes.flatten()
    
    for idx, (mk, ml) in enumerate(OUTPUT_METRICS):
        ax = axes_flat[idx]
        
        sub = sens_df[sens_df['Metric'] == mk].sort_values(
            'Sensitivity_Pct', ascending=True
        )
        
        bv = baseline[mk]
        y = np.arange(len(sub))
        
        lo = (sub['Min'].values - bv) / (abs(bv) + 1e-10) * 100
        hi = (sub['Max'].values - bv) / (abs(bv) + 1e-10) * 100
        
        ax.barh(y, hi, color='lightcoral', edgecolor='black', lw=1.2,
                label='증가 (Max)', alpha=0.8)
        ax.barh(y, lo, color='lightblue', edgecolor='black', lw=1.2,
                label='감소 (Min)', alpha=0.8)
        ax.axvline(0, color='black', lw=2, ls='--', alpha=0.7)
        
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f'{i+1}. {v}' for i, v in enumerate(sub['Variable'])],
            fontsize=10
        )
        ax.set_xlabel('기준 대비 변화율 (%)', fontsize=11)
        
        if 'Mult' in mk:
            bv_str = f'{bv:.2f}x'
        else:
            bv_str = f'{bv:,.0f}억'
        
        ax.set_title(f'{ml}\n(기준: {bv_str})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('sensitivity_tornado_tail.png', dpi=150, bbox_inches='tight')
    print("✅ Tornado 저장: sensitivity_tornado_tail.png")


def plot_curves(results, baseline, variables):
    """민감도 곡선"""
    n_vars = len(variables)
    n_met = len(OUTPUT_METRICS)
    
    fig, axes = plt.subplots(n_met, n_vars, 
                            figsize=(5*n_vars, 4*n_met), 
                            squeeze=False)
    fig.suptitle('민감도 곡선 - Crisis 시나리오',
                 fontsize=14, fontweight='bold', y=1.0)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_vars))
    
    for ci, (vn, vi) in enumerate(variables.items()):
        vr = results[vn]
        xv = [r['value'] for r in vr]
        
        for ri, (mk, ml) in enumerate(OUTPUT_METRICS):
            ax = axes[ri, ci]
            yv = [r[mk] for r in vr]
            bv = baseline[mk]
            
            ax.plot(xv, yv, 'o-', lw=2, ms=5, color=colors[ci])
            ax.axvline(vi['baseline'], color='red', ls=':', alpha=0.6, lw=1.5)
            ax.axhline(bv, color='gray', ls=':', alpha=0.4)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(vi['label'], fontsize=9)
            
            if ci == 0:
                ax.set_ylabel(ml, fontsize=9)
            
            ticks = xv if len(xv) <= 7 else xv[::2]
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [vi['format'](v) for v in ticks],
                rotation=45, fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig('sensitivity_curves_tail.png', dpi=150, bbox_inches='tight')
    print("✅ 곡선 저장: sensitivity_curves_tail.png")


def export_excel(sens_df, results, baseline, variables):
    """Excel 내보내기"""
    with pd.ExcelWriter('sensitivity_analysis_tail.xlsx', engine='openpyxl') as w:
        # Summary
        pivot = sens_df.pivot_table(
            index='Variable', columns='Metric', values='Sensitivity_Pct'
        )
        pivot.to_excel(w, sheet_name='Summary')
        
        # Detail
        sens_df.to_excel(w, sheet_name='Detail', index=False)
        
        # Baseline
        pd.DataFrame([baseline]).to_excel(w, sheet_name='Baseline', index=False)
        
        # Per variable
        for vn, vr in results.items():
            df = pd.DataFrame(vr)
            sheet = vn[:31]
            df.to_excel(w, sheet_name=sheet, index=False)
    
    print("✅ Excel 저장: sensitivity_analysis_tail.xlsx")


def print_insights(sens_df):
    """핵심 인사이트"""
    print("\n" + "="*80)
    print("📊 민감도 분석 핵심 인사이트")
    print("="*80)
    
    for mk, ml in OUTPUT_METRICS:
        sub = sens_df[sens_df['Metric'] == mk]
        top3 = sub.nlargest(3, 'Sensitivity_Pct')
        
        print(f"\n{'='*60}")
        print(f"🎯 {ml} — Top 3")
        print(f"{'='*60}")
        
        for i, row in enumerate(top3.itertuples(), 1):
            emoji = ['🥇', '🥈', '🥉'][i-1]
            
            if 'Mult' in mk:
                rng_str = f"{row.Min:.2f}x ~ {row.Max:.2f}x"
            else:
                rng_str = f"{row.Min:,.0f}억 ~ {row.Max:,.0f}억"
            
            print(f"\n{emoji} {i}위: {row.Variable}")
            print(f"   민감도: ±{row.Sensitivity_Pct:.1f}%")
            print(f"   범위:   {rng_str}")
            
            if row.Sensitivity_Pct > 50:
                tag = "⚠️  매우 높음 - 필수 관리"
            elif row.Sensitivity_Pct > 30:
                tag = "⚠️  높음 - 중요 변수"
            elif row.Sensitivity_Pct > 15:
                tag = "✓ 중간"
            else:
                tag = "✓ 낮음"
            print(f"   {tag}")
    
    print(f"\n{'='*60}")
    print("💡 핵심 발견")
    print(f"{'='*60}")
    
    # 공통 Top 변수
    all_top = []
    for mk, _ in OUTPUT_METRICS:
        sub = sens_df[sens_df['Metric'] == mk]
        all_top.extend(sub.nlargest(3, 'Sensitivity_Pct')['Variable'].tolist())
    
    from collections import Counter
    counts = Counter(all_top)
    
    print("\n전체 공통 핵심 변수:")
    for v, c in counts.most_common(3):
        print(f"  {v}: {c}개 지표에서 Top 3")
    
    print("\n⚠️  Tail Risk 특이사항:")
    tail_sens = sens_df[sens_df['Metric']=='Retail_Tail_Mult']
    if len(tail_sens) > 0:
        top_tail = tail_sens.nlargest(1, 'Sensitivity_Pct').iloc[0]
        print(f"  Tail 배율이 가장 민감한 변수: {top_tail['Variable']}")
        print(f"  민감도: ±{top_tail['Sensitivity_Pct']:.1f}%")
        print(f"  범위: {top_tail['Min']:.2f}x ~ {top_tail['Max']:.2f}x")


def main():
    """메인 실행"""
    print("="*80)
    print("민감도 분석 - Tail Risk 포함")
    print("시나리오: Crisis (41%), STO 28%")
    print("="*80)
    
    t0 = time.time()
    
    # 분석 실행
    results, baseline = run_oat_analysis(BASELINE_PARAMS, SENSITIVITY_VARIABLES)
    
    # 민감도 지표
    sens_df = calc_sensitivity_df(results, baseline, SENSITIVITY_VARIABLES)
    
    # 시각화
    plot_tornado(sens_df, baseline)
    plot_curves(results, baseline, SENSITIVITY_VARIABLES)
    
    # Excel
    export_excel(sens_df, results, baseline, SENSITIVITY_VARIABLES)
    
    # 인사이트
    print_insights(sens_df)
    
    elapsed = time.time() - t0
    print(f"\n⏱️  총 소요: {elapsed/60:.1f}분")
    print("="*80)
    print("✅ 분석 완료!")
    print("="*80)
    print("\n출력 파일:")
    print("  1. sensitivity_tornado_tail.png")
    print("  2. sensitivity_curves_tail.png")
    print("  3. sensitivity_analysis_tail.xlsx")


if __name__ == '__main__':
    main()