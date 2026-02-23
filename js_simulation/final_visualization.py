"""종합 시각화 로직
"""
from __future__ import annotations

from typing import Mapping, Sequence
import os

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

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEATMAP_VMIN = 0
HEATMAP_VMAX = 250000
HEATMAP_MARKETS = ['Perfect (100%)', 'Good (84%)', 'Recession (65%)', 'Crisis (41%)', 'Extreme (15%)']


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
    return pivot.reindex(columns=columns)


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

    for market_key in ['Perfect', 'Extreme']:
        market_label = market_scenarios[market_key]['label']
        row_var = [f'{market_label} - Fin VaR95']
        row_ext = [f'{market_label} - Ext VaR95']
        row_change = [f'{market_label} - Risk Δ', '0억']
        row_retail = [f'{market_label} - Retail Loss', 'N/A']

        for sto_label in sto_labels:
            scenario = f'{sto_label}_{market_key}'
            metric = all_results[scenario]['metrics']
            fin_var = metric['VaR_95']
            row_var.append(f'{fin_var:,.0f}억')

            if sto_label == 'Trad PF':
                ext_var = fin_var
            else:
                ext_var = fin_var + metric['retail_VaR_95']
            row_ext.append(f'{ext_var:,.0f}억')

            if sto_label != 'Trad PF':
                change = metric['retail_VaR_95']
                row_change.append(f'+{change:,.0f}억')
                row_retail.append(f'{change:,.0f}억')

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


def create_comprehensive_visualization(all_results: Mapping[str, dict], df: pd.DataFrame,
                                       sto_labels: Sequence[str], market_scenarios: Mapping[str, dict],
                                       reference_sto_label: str) -> None:
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
        trad_var = df[(df['Market'] == label) & (df['STO_Ratio'] == 'Trad PF')]['Financial_VaR95'].values[0]
        for sto_label in sto_labels[1:]:
            sto_var = df[(df['Market'] == label) & (df['STO_Ratio'] == sto_label)]['Financial_VaR95'].values[0]
            reduction = (trad_var - sto_var) / trad_var * 100
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

    plt.suptitle('Comprehensive Multi-Scenario Analysis: System Risk Analysis (V=1000억)',
                 fontsize=20, fontweight='bold', y=0.998)
    plt.savefig('comprehensive_analysis_v1000.png', dpi=150, bbox_inches='tight')
    print('\n✅ Comprehensive visualization saved: comprehensive_analysis_v1000.png')


def _load_from_excel(path: str):
    df = pd.read_excel(path, sheet_name='Summary')
    system = pd.read_excel(path, sheet_name='System_Risk_Analysis')
    # rebuild all_results minimally
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
            'results': {
                'losses': {
                    'systemic_loss': np.zeros((1, 1)),
                }
            },
            'sto_label': row['STO_Ratio'],
            'sto_ratio': 0,
            'market_label': row['Market'],
        }
    return all_results, df


def main() -> None:
    import os

    excel_path = os.path.join(os.path.dirname(__file__), 'simulation_results_v1000.xlsx')
    if not os.path.exists(excel_path):
        raise SystemExit('simulation_results_v1000.xlsx가 존재하지 않습니다. final_comparison.py를 먼저 실행하세요.')

    all_results, df = _load_from_excel(excel_path)
    create_comprehensive_visualization(
        all_results,
        df,
        STO_LABELS,
        MARKET_SCENARIOS,
        REFERENCE_STO_LABEL
    )


if __name__ == '__main__':
    main()
