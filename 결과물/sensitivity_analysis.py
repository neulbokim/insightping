"""
민감도 분석 (Sensitivity Analysis)
- 시나리오: Crisis 고정, Baseline STO 30%
- sto_ratio: 0~50% 범위로 일반 민감도 변수로 포함
- 출력 1: Tornado 2×3 (모든 변수 포함, VaR95 위 / VaR99 아래)
- 출력 2: STO 곡선 2×3 (x=STO ratio, y=VaR, Crisis 고정)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pf_simulation_v2 import SimulationParams, ImprovedPFSimulation
from pf_analysis_v2 import ImprovedRiskAnalyzer

import matplotlib.font_manager as fm
font_list = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in font_list:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'AppleGothic' in font_list:
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ===== 출력 지표: 2행×3열 =====
OUTPUT_METRICS_GRID = [
    [('Financial_VaR95', '금융기관 VaR95'),
     ('Retail_VaR95',    '개인 VaR95'),
     ('Extended_VaR95',  '확장 시스템 VaR95')],
    [('Financial_VaR99', '금융기관 VaR99'),
     ('Retail_VaR99',    '개인 VaR99'),
     ('Extended_VaR99',  '확장 시스템 VaR99')],
]
OUTPUT_METRICS_FLAT = [m for row in OUTPUT_METRICS_GRID for m in row]


# ===== Crisis 기준 파라미터 =====
BASE_PARAMS = {
    'n_simulations': 5000,
    'n_projects': 100,
    'T': 16,
    'sto_ratio': 0.30,            # Baseline STO
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
        'label': 'STO 발행 비율',
        'range': list(np.round(np.arange(0.0, 0.51, 0.05), 2)),
        'format': lambda x: f'{x*100:.0f}%',
    },
    'mu_sales_base': {
        'label': '분양률 성장률',
        'range': [-0.05, -0.02, 0.00, 0.02, 0.05],
        'format': lambda x: f'{x*100:+.1f}%',
    },
    'sigma_sales': {
        'label': '분양률 변동성',
        'range': [0.15, 0.20, 0.25, 0.30, 0.35],
        'format': lambda x: f'{x:.2f}',
    },
    'recovery_rate_base': {
        'label': '기본 회수율',
        'range': [0.10, 0.20, 0.25, 0.30, 0.40],
        'format': lambda x: f'{x*100:.0f}%',
    },
    'collateral_ratio': {
        'label': '담보 비율',
        'range': [0.10, 0.20, 0.30, 0.40, 0.50],
        'format': lambda x: f'{x*100:.0f}%',
    },
    'rho_base': {
        'label': '기본 상관계수',
        'range': [0.10, 0.20, 0.30, 0.40, 0.50],
        'format': lambda x: f'{x:.2f}',
    },
    'fire_sale_base': {
        'label': '급매각 할인율',
        'range': [0.30, 0.40, 0.50, 0.60, 0.70],
        'format': lambda x: f'{x*100:.0f}%',
    },
}


def run_single(params_dict: dict, seed: int = 42) -> dict:
    """단일 시뮬레이션 → 6개 지표 반환
    seed: baseline과 OAT 실험 모두 동일한 seed 사용 (Common Random Numbers 기법)
          → 파라미터 변화에 의한 차이만 측정, MC 노이즈 제거
    """
    np.random.seed(seed)
    params = SimulationParams(**params_dict)
    use_sto = params.sto_ratio > 0
    sim = ImprovedPFSimulation(params, use_sto=use_sto)
    results = sim.run_simulation()

    analyzer = ImprovedRiskAnalyzer(results, params)
    metrics = analyzer.calculate_all_metrics()

    fin95 = metrics['VaR_95']
    fin99 = metrics['VaR_99']

    if use_sto:
        retail_total = results['losses']['retail_loss'][:, :, -1].sum(axis=1)
        ret95 = float(np.percentile(retail_total, 95))
        ret99 = float(np.percentile(retail_total, 99))
    else:
        ret95 = ret99 = 0.0

    del results
    return {
        'Financial_VaR95': fin95,
        'Financial_VaR99': fin99,
        'Retail_VaR95':    ret95,
        'Retail_VaR99':    ret99,
        'Extended_VaR95':  fin95 + ret95,
        'Extended_VaR99':  fin99 + ret99,
    }


def run_oat() -> tuple:
    """One-at-a-Time 분석"""
    print("\n[Baseline] 계산...")
    baseline_res = run_single(BASE_PARAMS, seed=42)
    for k, v in baseline_res.items():
        print(f"  {k}: {v:,.0f}억")

    oat_results = {}
    total = sum(len(info['range']) for info in SENSITIVITY_VARIABLES.values())
    done = 0

    for var_name, info in SENSITIVITY_VARIABLES.items():
        print(f"\n  [{info['label']}]")
        rows = []
        for val in info['range']:
            res = run_single({**BASE_PARAMS, var_name: val}, seed=42)
            res['value'] = val
            rows.append(res)
            done += 1
            print(f"    {info['format'](val):>8} → "
                  f"Fin95:{res['Financial_VaR95']:>7,.0f}  "
                  f"Ret95:{res['Retail_VaR95']:>7,.0f}  "
                  f"Ext95:{res['Extended_VaR95']:>7,.0f}  [{done}/{total}]")
        oat_results[var_name] = rows

    return oat_results, baseline_res


def calc_sensitivity_df(oat_results: dict, baseline: dict) -> pd.DataFrame:
    rows = []
    for var_name, var_rows in oat_results.items():
        label = SENSITIVITY_VARIABLES[var_name]['label']
        for mk, ml in OUTPUT_METRICS_FLAT:
            vals = [r[mk] for r in var_rows]
            base_val = baseline[mk]
            rng = max(vals) - min(vals)
            pct = (rng / abs(base_val) * 100) if base_val != 0 else 0.0
            rows.append({
                'Variable':        label,
                'Metric':          mk,
                'Metric_Label':    ml,
                'Sensitivity_Pct': pct,
                'Min':    min(vals),
                'Max':    max(vals),
                'Range':  rng,
                'Baseline': base_val,
            })
    return pd.DataFrame(rows)


def plot_tornado(sens_df: pd.DataFrame, baseline: dict) -> None:
    """Tornado 2행×3열 — 모든 민감도 변수 포함"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(
        f'Tornado 다이어그램 — Crisis 시나리오 (Baseline STO {BASE_PARAMS["sto_ratio"]*100:.0f}%)',
        fontsize=16, fontweight='bold', y=1.01
    )

    for ri, row_metrics in enumerate(OUTPUT_METRICS_GRID):
        for ci, (mk, ml) in enumerate(row_metrics):
            ax = axes[ri, ci]
            sub = (sens_df[sens_df['Metric'] == mk]
                   .sort_values('Sensitivity_Pct', ascending=True))

            bv = baseline[mk]
            y = np.arange(len(sub))
            lo = (sub['Min'].values - bv) / (abs(bv) + 1e-10) * 100
            hi = (sub['Max'].values - bv) / (abs(bv) + 1e-10) * 100

            ax.barh(y, hi, color='lightcoral', edgecolor='black', lw=0.8,
                    label='증가 (Max)', alpha=0.85)
            ax.barh(y, lo, color='lightblue',  edgecolor='black', lw=0.8,
                    label='감소 (Min)', alpha=0.85)
            ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)

            ax.set_yticks(y)
            ax.set_yticklabels(
                [f'{i+1}. {v}' for i, v in enumerate(sub['Variable'])],
                fontsize=10
            )
            ax.set_xlabel('기준 대비 변화율 (%)', fontsize=10)
            ax.set_title(f'{ml}\n(기준: {bv:,.0f}억)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('sensitivity_tornado.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 저장: sensitivity_tornado.png")


def plot_sto_curve(oat_results: dict, baseline: dict) -> None:
    """STO 곡선 2행×3열 — x=STO ratio, y=VaR (Crisis 고정)"""
    sto_rows = oat_results['sto_ratio']
    x_vals = [r['value'] * 100 for r in sto_rows]  # %로 변환
    baseline_sto_pct = BASE_PARAMS['sto_ratio'] * 100

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        'STO 비율별 리스크 변화 — Crisis 시나리오 (다른 변수 기준값 고정)',
        fontsize=16, fontweight='bold', y=1.01
    )

    colors = {'VaR95': '#1f77b4', 'VaR99': '#d62728'}

    for ri, row_metrics in enumerate(OUTPUT_METRICS_GRID):
        for ci, (mk, ml) in enumerate(row_metrics):
            ax = axes[ri, ci]
            y_vals = [r[mk] for r in sto_rows]
            bv = baseline[mk]

            level = 'VaR99' if '99' in mk else 'VaR95'
            ax.plot(x_vals, y_vals, 'o-', lw=2.5, ms=6,
                    color=colors[level], label=ml)
            ax.axvline(baseline_sto_pct, color='gray', ls=':', lw=1.5,
                       label=f'Baseline {baseline_sto_pct:.0f}%')
            ax.axhline(bv, color='gray', ls='--', lw=1.0, alpha=0.5)

            ax.set_xlabel('STO 발행 비율 (%)', fontsize=11)
            ax.set_ylabel('VaR (억원)', fontsize=11)
            ax.set_title(f'{ml}\n(Baseline: {bv:,.0f}억)', fontsize=12, fontweight='bold')
            ax.set_xticks(x_vals)
            ax.set_xticklabels([f'{v:.0f}%' for v in x_vals], rotation=45, fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensitivity_sto_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 저장: sensitivity_sto_curve.png")


def export_excel(oat_results: dict, baseline: dict, sens_df: pd.DataFrame) -> None:
    fname = 'sensitivity_analysis.xlsx'
    with pd.ExcelWriter(fname, engine='openpyxl') as w:
        # Summary pivot
        pivot = sens_df.pivot_table(
            index='Variable', columns='Metric', values='Sensitivity_Pct'
        )
        pivot.to_excel(w, sheet_name='Summary')
        sens_df.to_excel(w, sheet_name='Detail', index=False)
        pd.DataFrame([baseline]).to_excel(w, sheet_name='Baseline', index=False)

        # STO curve raw data
        sto_df = pd.DataFrame(oat_results['sto_ratio'])
        sto_df.to_excel(w, sheet_name='STO_Curve', index=False)

        # Per-variable sheets
        for var_name, rows in oat_results.items():
            label = SENSITIVITY_VARIABLES[var_name]['label'][:20]
            pd.DataFrame(rows).to_excel(w, sheet_name=label, index=False)

    print(f"✅ Excel 저장: {fname}")


def main():
    print("="*80)
    print("민감도 분석 — Crisis 시나리오, Baseline STO 30%")
    print("sto_ratio: 0~50% 범위 포함")
    print("="*80)

    t0 = time.time()

    oat_results, baseline = run_oat()
    sens_df = calc_sensitivity_df(oat_results, baseline)

    plot_tornado(sens_df, baseline)
    plot_sto_curve(oat_results, baseline)
    export_excel(oat_results, baseline, sens_df)

    print(f"\n⏱️  총 소요: {(time.time()-t0)/60:.1f}분")
    print("="*80)
    print("✅ 완료! 출력 파일:")
    print("  sensitivity_tornado.png   (Tornado 2×3)")
    print("  sensitivity_sto_curve.png (STO 곡선 2×3)")
    print("  sensitivity_analysis.xlsx")


if __name__ == '__main__':
    main()
