"""종합 시각화 로직
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Mapping, Sequence

STO_LABELS = ['Trad PF', 'STO 10%', 'STO 20%', 'STO 30%', 'STO 40%',
              'STO 50%', 'STO 60%', 'STO 70%', 'STO 80%', 'STO 90%', 'STO 100%']
REFERENCE_STO_LABEL = 'STO 100%'
MARKET_SCENARIOS = {
    'Perfect': {'label': 'Perfect (100%)', 'color': 'darkgreen'},
    'Good': {'label': 'Good (84%)', 'color': 'lightgreen'},
    'Recession': {'label': 'Recession (65%)', 'color': 'orange'},
    'Crisis': {'label': 'Crisis (41%)', 'color': 'red'},
    'Extreme': {'label': 'Extreme (15%)', 'color': 'darkred'},
}

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEATMAP_VMIN = 0
HEATMAP_VMAX = 250000
HEATMAP_MARKETS = ['Perfect (100%)', 'Good (84%)', 'Recession (65%)', 'Crisis (41%)', 'Extreme (15%)']


def _suffix_from_params(params):
    return f"v{int(params.total_project_value)}_n{params.n_simulations}"


def _set_korean_font() -> None:
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'Malgun Gothic' in font_list:
        font_name = 'Malgun Gothic'
    elif 'NanumGothic' in font_list:
        font_name = 'NanumGothic'
    elif 'AppleGothic' in font_list:
        font_name = 'AppleGothic'
    else:
        font_name = 'DejaVu Sans'

    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10


def _format_currency(value: float) -> str:
    return f'{value:,.0f}억'


def _pivot_heatmap(df: pd.DataFrame, value_col: str, columns: Sequence[str]) -> pd.DataFrame:
    pivot = df.pivot(index='Market', columns='STO_Ratio', values=value_col)
    pivot = pivot.reindex(HEATMAP_MARKETS)
    pivot = pivot.reindex(columns=columns)
    return pivot.fillna(0.0)


def _format_cell(value: float, fmt: str | None) -> str:
    if fmt == 'percent':
        return f'{value:.1f}%'
    if fmt == 'change':
        return f'+{value:,.0f}억'
    return _format_currency(value)


def _plot_heapmap(ax, data: pd.DataFrame, title: str, cbar_label: str, vmin: float, vmax: float,
                  fmt: str | None = None) -> None:
    im = ax.imshow(data.values, cmap='Reds', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.columns, fontsize=11)
    ax.set_yticklabels(data.index, fontsize=11)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iat[i, j]
            if np.isnan(val):
                label = 'N/A'
            else:
                label = _format_cell(val, fmt)
            ax.text(j, i, label, ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)


def _stacked_table(ax, sto_labels: Sequence[str], all_results: Mapping[str, dict],
                   market_scenarios: Mapping[str, dict]) -> None:
    rows: list[list[str]] = []
    header = ['Metric'] + list(sto_labels)
    rows.append(header)

    def _get_metric_value(scenario_key: str, metric_key: str, default: float = 0.0) -> float:
        return all_results.get(scenario_key, {}).get('metrics', {}).get(metric_key, default)

    for market_key in ['Perfect', 'Extreme']:
        market_label = market_scenarios[market_key]['label']
        row_var = [f'{market_label} - Fin VaR95']
        row_ext = [f'{market_label} - Ext VaR95']
        row_change = [f'{market_label} - Risk Δ']
        row_retail = [f'{market_label} - Retail Loss']

        for sto_label in sto_labels:
            scenario = f'{sto_label}_{market_key}'
            fin_var = _get_metric_value(scenario, 'VaR_95')
            retail_var = _get_metric_value(scenario, 'retail_VaR_95')

            row_var.append(f'{fin_var:,.0f}억')
            ext_var = fin_var if sto_label == 'Trad PF' else fin_var + retail_var
            row_ext.append(f'{ext_var:,.0f}억')
            row_change.append(f'+{retail_var:,.0f}억')
            row_retail.append(f'{retail_var:,.0f}억')

        rows.extend([row_var, row_ext, row_change, row_retail])

    table = ax.table(cellText=rows, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    num_columns = len(sto_labels) + 1

    for col in range(num_columns):
        table[(0, col)].set_facecolor('#4472C4')
        table[(0, col)].set_text_props(weight='bold', color='white')

    for idx in range(1, len(rows)):
        for col in range(num_columns):
            if idx % 4 == 0:
                table[(idx, col)].set_facecolor('#E7E6E6')
            elif 'Risk Δ' in rows[idx][0]:
                table[(idx, col)].set_facecolor('#FFF2CC')


def _safe_financial_var(df: pd.DataFrame, market_label: str, sto_label: str) -> float:
    mask = (df['Market'] == market_label) & (df['STO_Ratio'] == sto_label)
    if mask.any():
        return float(df.loc[mask, 'Financial_VaR95'].iloc[0])
    return 0.0


def create_comprehensive_visualization(all_results: Mapping[str, dict], df: pd.DataFrame,
                                       sto_labels: Sequence[str], market_scenarios: Mapping[str, dict],
                                       reference_sto_label: str, params: SimpleNamespace) -> None:
    _set_korean_font()

    fig = plt.figure(figsize=(35, 28), dpi=100)
    gs = fig.add_gridspec(6, 5, hspace=0.6, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, :3])
    pivot_financial = _pivot_heatmap(df, 'Financial_VaR95', sto_labels)
    _plot_heapmap(ax1, pivot_financial, '금융기관 VaR95 (억원)', 'VaR95 (억원)',
                  HEATMAP_VMIN, HEATMAP_VMAX)

    ax2 = fig.add_subplot(gs[0, 3:])
    pivot_extended = _pivot_heatmap(df, 'Extended_Systemic_VaR95', sto_labels)
    _plot_heapmap(ax2, pivot_extended, '확장 시스템 리스크 VaR95\n(금융+개인, 억원)', 'VaR95 (억원)',
                  HEATMAP_VMIN, HEATMAP_VMAX)

    ax3 = fig.add_subplot(gs[1, :3])
    pivot_change = _pivot_heatmap(df[df['STO_Ratio'] != 'Trad PF'], 'System_Risk_Change', sto_labels[1:])
    _plot_heapmap(ax3, pivot_change, '개인 투자자 추가로 인한\n시스템 리스크 증가 (억원)', '증가액 (억원)',
                  HEATMAP_VMIN, max(np.nanmax(pivot_change.values), 1), fmt='change')

    ax4 = fig.add_subplot(gs[1, 3:])
    pivot_retail = _pivot_heatmap(df[df['STO_Ratio'] != 'Trad PF'], 'Retail_VaR95_Absolute', sto_labels[1:])
    _plot_heapmap(ax4, pivot_retail, 'STO 비율별 개인 투자자 손실 VaR95 (억원)', '손실액 (억원)', 0,
                  pivot_retail.values.max())

    market_order = ['Perfect', 'Good', 'Recession', 'Crisis', 'Extreme']
    for idx, market_key in enumerate(market_order):
        ax = fig.add_subplot(gs[2, idx])
        label = market_scenarios[market_key]['label']
        color = market_scenarios[market_key]['color']
        data = df[df['Market'] == label]
        bars = ax.bar(data['STO_Ratio'], data['Financial_VaR95'], color=color, edgecolor='black', alpha=0.7, width=0.6)
        ax.set_ylabel('VaR95 (억원)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height / 1000)}K',
                        ha='center', va='bottom', fontsize=7)

    ax5 = fig.add_subplot(gs[3, :3])
    reductions = []
    for market_key in market_scenarios:
        label = market_scenarios[market_key]['label']
        trad_var = _safe_financial_var(df, label, 'Trad PF')
        for sto_label in sto_labels[1:]:
            sto_var = _safe_financial_var(df, label, sto_label)
            reduction = (trad_var - sto_var) / trad_var * 100 if trad_var != 0 else 0.0
            reductions.append({'Market': label, 'STO_Ratio': sto_label, 'Reduction': reduction})

    pivot_reduction = pd.DataFrame(reductions).pivot(index='Market', columns='STO_Ratio', values='Reduction')
    pivot_reduction = pivot_reduction.reindex(HEATMAP_MARKETS)
    pivot_reduction = pivot_reduction.reindex(columns=sto_labels[1:])
    _plot_heapmap(ax5, pivot_reduction, 'STO 도입 효과: 금융기관 VaR 감소율 vs 기존 PF (%)', '감소율 (%)', 0, 100, fmt='percent')

    ax6 = fig.add_subplot(gs[3, 3:])
    time_data = []
    for market_key in market_scenarios:
        label = market_scenarios[market_key]['label']
        for sto_label in sto_labels:
            scenario = f"{sto_label}_{market_key}"
            if scenario not in all_results:
                continue
            sys_loss = all_results[scenario]['results']['losses']['systemic_loss'].mean(axis=0)
            threshold = 100 * 1000 * 0.01
            time_idx = np.where(sys_loss > threshold)[0]
            time_data.append({'Market': label, 'STO': sto_label, 'Time': time_idx[0] if len(time_idx) else 16})

    pivot_time = pd.DataFrame(time_data).pivot(index='Market', columns='STO', values='Time')
    pivot_time = pivot_time.reindex(HEATMAP_MARKETS)
    ax8 = fig.add_subplot(gs[4, 3:])
    im6 = ax8.imshow(pivot_time.values, cmap='Reds', aspect='auto')
    ax8.set_xticks(np.arange(len(pivot_time.columns)))
    ax8.set_yticks(np.arange(len(pivot_time.index)))
    ax8.set_xticklabels(pivot_time.columns, fontsize=9)
    ax8.set_yticklabels(pivot_time.index)
    for i in range(len(pivot_time.index)):
        for j in range(len(pivot_time.columns)):
            ax8.text(j, i, f'{pivot_time.iat[i, j]:.1f}Q', ha='center', va='center', color='black', fontsize=9,
                     fontweight='bold')
    ax8.set_title('포트폴리오 1% 손실 도달 시간 (분기)', fontsize=15, fontweight='bold', pad=15)
    cbar8 = plt.colorbar(im6, ax=ax8)
    cbar8.set_label('시간 (분기)', fontsize=11)

    ax9 = fig.add_subplot(gs[5, :])
    ax9.axis('off')
    _stacked_table(ax9, sto_labels, all_results, market_scenarios)

    suffix = _suffix_from_params(params)
    plt.suptitle(
        f'Comprehensive Multi-Scenario Analysis: System Risk Analysis (V={int(params.total_project_value)}억, n={params.n_simulations})',
        fontsize=20, fontweight='bold', y=0.998
    )
    filename = f'comprehensive_analysis_{suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'\n✅ Comprehensive visualization saved: {filename}')

    # ✅ 분포 패널(시스템/개인/확장) 생성
    plot_distribution_panels(all_results, market_scenarios, sto_labels, params)

def plot_distribution_panels(all_results, market_scenarios, sto_labels, params: SimpleNamespace):
    """
    한 PNG에 system/retail/extended를 모두 포함:
      - STO별: 3행(시스템/개인/확장) x 10열(STO10~100)
      - 시장별: 3행(시스템/개인/확장) x 5열(Perfect~Extreme)
    """
    percentiles = [0, 5, 25, 50, 75, 95, 99]
    _set_korean_font()

    market_order = ['Perfect', 'Good', 'Recession', 'Crisis', 'Extreme']
    sto_labels_10 = [s for s in sto_labels if s != 'Trad PF']  # 10개
    suffix = _suffix_from_params(params)

    # (dist_key, row_title, y_label, file_tag)
    dist_specs = [
        ('systemic_percentiles', 'SYSTEM',   '손실 (억원)'),
        ('retail_percentiles',   'RETAIL',   '손실 (억원)'),
        ('extended_percentiles', 'EXTENDED', '손실 (억원)'),
    ]

    def _global_ymax(dist_key: str) -> float:
        mx = 0.0
        for mk in market_scenarios.keys():
            for sl in sto_labels:
                scenario = f"{sl}_{mk}"
                dist = all_results.get(scenario, {}).get('metrics', {}).get(dist_key, {})
                if dist:
                    mx = max(mx, max(dist.values()))
        return mx if mx > 0 else 1.0

    # 타입별 y축 스케일(각 타입끼리 통일)
    y_lims = {k: _global_ymax(k) * 1.05 for k, _, _ in dist_specs}

    # =========================================================
    # 1) STO별 패널: 3행 x 10열 (각 열=STO, 각 행=system/retail/extended)
    # =========================================================
    fig1, axes1 = plt.subplots(
        len(dist_specs), len(sto_labels_10),
        figsize=(3.0 * len(sto_labels_10), 3.0 * len(dist_specs)),
        dpi=140,
        sharex=True
    )

    # axes1 shape 보정(혹시 1차원으로 나오는 경우 대비)
    axes1 = np.atleast_2d(axes1)

    # 범례는 그림 전체에 1번만(시장 라인)
    legend_handles = None
    legend_labels = None

    for r, (dist_key, row_title, ylab) in enumerate(dist_specs):
        for c, sto_label in enumerate(sto_labels_10):
            ax = axes1[r, c]

            for market_key in market_order:
                market_label = market_scenarios[market_key]['label']
                market_color = market_scenarios[market_key]['color']

                scenario = f"{sto_label}_{market_key}"
                dist = all_results.get(scenario, {}).get('metrics', {}).get(dist_key, {})

                xs, ys = [], []
                for p in percentiles:
                    if p in dist:
                        xs.append(p)
                        ys.append(dist[p])

                if xs:
                    ax.plot(xs, ys, marker='o', linewidth=2, color=market_color, label=market_label)

            ax.grid(True, alpha=0.25)
            ax.set_ylim(0, y_lims[dist_key])
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            if r == 0:
                ax.set_title(sto_label, fontsize=10, fontweight='bold')

            if c == 0:
                ax.set_ylabel(f"{row_title}\n{ylab}", fontsize=9, fontweight='bold')

            if r == len(dist_specs) - 1:
                ax.set_xlabel('Percentile', fontsize=9)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig1.suptitle(
        f"Distribution Panels by STO (V={int(params.total_project_value)}억, n={params.n_simulations})",
        fontsize=14, fontweight='bold', y=1.02
    )
    if legend_handles:
        fig1.legend(
            legend_handles, legend_labels,
            loc='lower center', ncol=5, fontsize=9,
            bbox_to_anchor=(0.5, -0.01),
            frameon=False
        )

    plt.tight_layout()
    filename1 = f"distribution_panels_by_sto_all_{suffix}.png"
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ STO 통합 패널 저장: {filename1}")

    # =========================================================
    # 2) 시장별 패널: 3행 x 5열 (각 열=시장, 각 행=system/retail/extended)
    # =========================================================
    fig2, axes2 = plt.subplots(
        len(dist_specs), len(market_order),
        figsize=(4.2 * len(market_order), 3.0 * len(dist_specs)),
        dpi=140,
        sharex=True
    )
    axes2 = np.atleast_2d(axes2)

    # STO 라인 범례(11개)는 별도로 한 번만
    sto_legend_handles = None
    sto_legend_labels = None

    for r, (dist_key, row_title, ylab) in enumerate(dist_specs):
        for c, market_key in enumerate(market_order):
            ax = axes2[r, c]
            market_label = market_scenarios[market_key]['label']

            for sto_label in sto_labels:  # Trad PF 포함
                scenario = f"{sto_label}_{market_key}"
                dist = all_results.get(scenario, {}).get('metrics', {}).get(dist_key, {})

                xs, ys = [], []
                for p in percentiles:
                    if p in dist:
                        xs.append(p)
                        ys.append(dist[p])

                if xs:
                    ax.plot(xs, ys, marker='o', linewidth=1.4, label=sto_label)

            ax.grid(True, alpha=0.25)
            ax.set_ylim(0, y_lims[dist_key])
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            if r == 0:
                ax.set_title(market_label, fontsize=10, fontweight='bold')

            if c == 0:
                ax.set_ylabel(f"{row_title}\n{ylab}", fontsize=9, fontweight='bold')

            if r == len(dist_specs) - 1:
                ax.set_xlabel('Percentile', fontsize=9)

            if sto_legend_handles is None:
                sto_legend_handles, sto_legend_labels = ax.get_legend_handles_labels()

    fig2.suptitle(
        f"Distribution Panels by Market (V={int(params.total_project_value)}억, n={params.n_simulations})",
        fontsize=14, fontweight='bold', y=1.02
    )
    if sto_legend_handles:
        fig2.legend(
            sto_legend_handles, sto_legend_labels,
            loc='lower center', ncol=6, fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False
        )

    plt.tight_layout()
    filename2 = f"distribution_panels_by_market_all_{suffix}.png"
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ 시장 통합 패널 저장: {filename2}")

def _attach_percentiles_from_distribution(all_results: dict, distribution_df: pd.DataFrame | None) -> None:
    """
    Distribution 시트의 Type(systemic/retail/extended)을 metrics에 각각 붙인다.
      - metrics['systemic_percentiles']
      - metrics['retail_percentiles']
      - metrics['extended_percentiles']
    """
    if distribution_df is None or distribution_df.empty:
        return

    type_map = {
        'systemic': 'systemic_percentiles',
        'retail': 'retail_percentiles',
        'extended': 'extended_percentiles',
    }

    for scenario, group in distribution_df.groupby('Scenario'):
        metrics = all_results.get(scenario, {}).get('metrics', {})
        if not metrics:
            continue

        for t, key in type_map.items():
            sub = group[group['Type'] == t]
            if sub.empty:
                continue

            pcts = sub['Percentile'].astype(int).tolist()
            vals = sub['Value'].astype(float).tolist()
            metrics[key] = dict(zip(pcts, vals))


def _load_from_excel(path: str):
    df = pd.read_excel(path, sheet_name='Summary')
    system = pd.read_excel(path, sheet_name='System_Risk_Analysis')
    try:
        distribution = pd.read_excel(path, sheet_name='Distribution')
    except ValueError:
        distribution = None

    all_results = {}
    for _, row in system.iterrows():
        scenario = row['Scenario']
        all_results[scenario] = {
            'metrics': {
                'VaR_95': row['Financial_VaR95'],
                'ES_95': row['Financial_ES95'],
                'retail_VaR_95': row.get('Retail_VaR95', 0),
                'extended_VaR_95': row.get('Extended_System_VaR95', row['Financial_VaR95']),
            },
            # 시각화용 더미 (time chart가 참조)
            'results': {
                'losses': {
                    'systemic_loss': np.zeros((1, 1)),
                }
            },
            'sto_label': row['STO_Ratio'],
            'sto_ratio': 0,
            'market_label': row['Market'],
        }

    _attach_percentiles_from_distribution(all_results, distribution)
    return all_results, df


def _resolve_excel_path(project_value: int, n_simulations: int, explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    base_name = f"results/simulation_results_v{project_value}_n{n_simulations}.xlsx"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, base_name),
        os.path.join(os.path.dirname(script_dir), base_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PF simulation visualization from saved Excel")
    parser.add_argument("--project-value", "-v", type=int, default=1000,
                        help="total project value (억원) used in the Excel filename pattern (default: 1000)")
    parser.add_argument("--n-simulations", "-n", type=int, default=1000,
                        help="per-scenario MC trials used in the Excel filename pattern (default: 1000)")
    parser.add_argument("--input-file", "-i", type=str,
                        help="경로를 지정하면 v/n 패턴 대신 해당 파일을 사용합니다.")
    args = parser.parse_args()

    excel_path = _resolve_excel_path(args.project_value, args.n_simulations, args.input_file)
    if not os.path.exists(excel_path):
        raise SystemExit(
            f'{os.path.basename(excel_path)}가 존재하지 않습니다. '
            f'final_comparison.py를 먼저 실행하거나 --input-file로 경로를 제공하세요.'
        )

    params = SimpleNamespace(total_project_value=args.project_value, n_simulations=args.n_simulations)
    all_results, df = _load_from_excel(excel_path)
    print(f"\n✅ Loaded simulation results: {os.path.basename(excel_path)}")

    create_comprehensive_visualization(
        all_results,
        df,
        STO_LABELS,
        MARKET_SCENARIOS,
        REFERENCE_STO_LABEL,
        params
    )


if __name__ == '__main__':
    main()