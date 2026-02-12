"""
부동산 PF 시나리오 분석 자동화
- STO 비율 × 시장 환경 = 12개 시나리오
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pf_simulation_v2 import SimulationParams, ImprovedPFSimulation
from pf_analysis_v2 import ImprovedRiskAnalyzer

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ========== 시나리오 정의 ==========

SCENARIOS = [
    # STO 15% - 보수적
    {'id': 1, 'name': 'STO15%-정상', 'sto_ratio': 0.15, 'mu': 0.04, 'sigma': 0.15, 'market': '정상'},
    {'id': 2, 'name': 'STO15%-침체', 'sto_ratio': 0.15, 'mu': 0.02, 'sigma': 0.20, 'market': '침체'},
    {'id': 3, 'name': 'STO15%-위기', 'sto_ratio': 0.15, 'mu': 0.00, 'sigma': 0.25, 'market': '위기'},
    
    # STO 28% - 기준
    {'id': 4, 'name': 'STO28%-정상 (기준)', 'sto_ratio': 0.28, 'mu': 0.04, 'sigma': 0.15, 'market': '정상'},
    {'id': 5, 'name': 'STO28%-침체', 'sto_ratio': 0.28, 'mu': 0.02, 'sigma': 0.20, 'market': '침체'},
    {'id': 6, 'name': 'STO28%-위기', 'sto_ratio': 0.28, 'mu': 0.00, 'sigma': 0.25, 'market': '위기'},
    
    # STO 40% - 적극적
    {'id': 7, 'name': 'STO40%-정상', 'sto_ratio': 0.40, 'mu': 0.04, 'sigma': 0.15, 'market': '정상'},
    {'id': 8, 'name': 'STO40%-침체', 'sto_ratio': 0.40, 'mu': 0.02, 'sigma': 0.20, 'market': '침체'},
    {'id': 9, 'name': 'STO40%-위기', 'sto_ratio': 0.40, 'mu': 0.00, 'sigma': 0.25, 'market': '위기'},
    
    # 기존 PF - 비교군
    {'id': 0, 'name': '기존PF-정상', 'sto_ratio': 0.00, 'mu': 0.04, 'sigma': 0.15, 'market': '정상'},
    {'id': 10, 'name': '기존PF-침체', 'sto_ratio': 0.00, 'mu': 0.02, 'sigma': 0.20, 'market': '침체'},
    {'id': 11, 'name': '기존PF-위기', 'sto_ratio': 0.00, 'mu': 0.00, 'sigma': 0.25, 'market': '위기'},
]


def run_scenario(scenario_config, n_projects=100, n_simulations=10000):
    """단일 시나리오 실행"""
    
    print(f"\n{'='*70}")
    print(f"시나리오 {scenario_config['id']}: {scenario_config['name']}")
    print(f"{'='*70}")
    print(f"  STO 비율: {scenario_config['sto_ratio']:.0%}")
    print(f"  분양률 증가: {scenario_config['mu']:.1%}/분기")
    print(f"  변동성: {scenario_config['sigma']:.0%}")
    
    # 파라미터 설정
    params = SimulationParams(
        n_projects=n_projects,
        T=16,
        n_simulations=n_simulations,
        sto_ratio=scenario_config['sto_ratio'],
        mu_sales_base=scenario_config['mu'],
        sigma_sales=scenario_config['sigma'],
    )
    
    # 시뮬레이션 실행
    use_sto = scenario_config['sto_ratio'] > 0
    
    start = time.time()
    sim = ImprovedPFSimulation(params, use_sto=use_sto)
    results = sim.run_simulation()
    elapsed = time.time() - start
    
    print(f"  ✓ 완료 (소요: {elapsed:.1f}초)")
    
    # 리스크 분석
    analyzer = ImprovedRiskAnalyzer(results, params)
    metrics = analyzer.calculate_all_metrics()
    
    return {
        'scenario': scenario_config,
        'results': results,
        'metrics': metrics,
        'elapsed': elapsed,
    }


def create_comparison_table(scenario_results):
    """비교 분석 표 생성"""
    
    data = []
    
    for sr in scenario_results:
        scenario = sr['scenario']
        metrics = sr['metrics']
        
        row = {
            'ID': scenario['id'],
            '시나리오': scenario['name'],
            'STO비율': f"{scenario['sto_ratio']:.0%}",
            '시장환경': scenario['market'],
            '금융VaR95': f"{metrics['VaR_95']:,.0f}",
            '금융ES95': f"{metrics['ES_95']:,.0f}",
            '회수율': f"{metrics['overall_recovery_rate']:.1%}",
            '동시부실10%': f"{metrics['prob_10pct_simultaneous']:.1%}",
            '전이시간': f"{metrics['mean_contagion_time']:.1f}",
            '전이확률': f"{metrics['prob_contagion']:.1%}",
        }
        
        # STO 추가 지표
        if 'retail_VaR_95' in metrics:
            row['개인VaR95'] = f"{metrics['retail_VaR_95']:,.0f}"
            row['개인손실률'] = f"{metrics['retail_loss_rate']:.2%}"
            row['확장VaR95'] = f"{metrics['extended_VaR_95']:,.0f}"
        else:
            row['개인VaR95'] = '-'
            row['개인손실률'] = '-'
            row['확장VaR95'] = '-'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_summary_table(scenario_results):
    """요약 표 생성 - 기존 PF 대비 변화율"""
    
    # 기존 PF 기준값 찾기
    baseline = None
    for sr in scenario_results:
        if sr['scenario']['id'] == 0:  # 기존PF-정상
            baseline = sr['metrics']
            break
    
    if baseline is None:
        print("⚠️ 기존 PF 기준 시나리오 없음")
        return None
    
    data = []
    
    for sr in scenario_results:
        scenario = sr['scenario']
        metrics = sr['metrics']
        
        # STO만 비교
        if scenario['sto_ratio'] == 0:
            continue
        
        # 변화율 계산
        fin_var_change = (metrics['VaR_95'] / baseline['VaR_95'] - 1) * 100
        
        row = {
            'ID': scenario['id'],
            '시나리오': scenario['name'],
            'STO비율': f"{scenario['sto_ratio']:.0%}",
            '시장': scenario['market'],
            '금융VaR': f"{metrics['VaR_95']:,.0f}억",
            '기존대비': f"{fin_var_change:+.1f}%",
        }
        
        if 'retail_VaR_95' in metrics:
            row['개인VaR'] = f"{metrics['retail_VaR_95']:,.0f}억"
            row['개인손실률'] = f"{metrics['retail_loss_rate']:.2%}"
            
            ext_change = (metrics['extended_VaR_95'] / baseline['VaR_95'] - 1) * 100
            row['확장VaR'] = f"{metrics['extended_VaR_95']:,.0f}억"
            row['확장대비'] = f"{ext_change:+.1f}%"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_scenario_comparison(scenario_results, save_path=None):
    """시나리오 비교 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 데이터 준비
    sto_scenarios = [sr for sr in scenario_results if sr['scenario']['sto_ratio'] > 0]
    baseline_scenarios = [sr for sr in scenario_results if sr['scenario']['sto_ratio'] == 0]
    
    # 1. STO 비율별 리스크 분해 (정상 시장)
    normal_scenarios = [sr for sr in sto_scenarios if sr['scenario']['market'] == '정상']
    normal_scenarios = sorted(normal_scenarios, key=lambda x: x['scenario']['sto_ratio'])
    
    sto_ratios = [sr['scenario']['sto_ratio'] * 100 for sr in normal_scenarios]
    fin_vars = [sr['metrics']['VaR_95'] for sr in normal_scenarios]
    retail_vars = [sr['metrics'].get('retail_VaR_95', 0) for sr in normal_scenarios]
    
    x = np.arange(len(sto_ratios))
    axes[0, 0].bar(x, fin_vars, label='금융권', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x, retail_vars, bottom=fin_vars, label='개인', color='coral', alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'{r:.0f}%' for r in sto_ratios])
    axes[0, 0].set_xlabel('STO 후순위 비율', fontsize=11)
    axes[0, 0].set_ylabel('VaR 95% (억원)', fontsize=11)
    axes[0, 0].set_title('STO 비율별 리스크 분해 (정상 시장)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # 2. 시장 환경별 민감도 (STO 28%)
    sto28_scenarios = [sr for sr in scenario_results if sr['scenario']['sto_ratio'] == 0.28]
    markets = ['정상', '침체', '위기']
    sto28_by_market = {sr['scenario']['market']: sr for sr in sto28_scenarios}
    
    baseline_by_market = {sr['scenario']['market']: sr for sr in baseline_scenarios}
    
    market_x = np.arange(len(markets))
    baseline_vars = [baseline_by_market[m]['metrics']['VaR_95'] for m in markets]
    fin_vars_28 = [sto28_by_market[m]['metrics']['VaR_95'] for m in markets]
    ext_vars_28 = [sto28_by_market[m]['metrics']['extended_VaR_95'] for m in markets]
    
    axes[0, 1].plot(market_x, baseline_vars, marker='o', linewidth=2, label='기존 PF', color='gray')
    axes[0, 1].plot(market_x, fin_vars_28, marker='s', linewidth=2, label='STO 금융권', color='steelblue')
    axes[0, 1].plot(market_x, ext_vars_28, marker='^', linewidth=2, label='STO 확장', color='coral')
    axes[0, 1].set_xticks(market_x)
    axes[0, 1].set_xticklabels(markets)
    axes[0, 1].set_xlabel('시장 환경', fontsize=11)
    axes[0, 1].set_ylabel('VaR 95% (억원)', fontsize=11)
    axes[0, 1].set_title('시장 환경별 민감도 (STO 28%)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 개인 투자자 리스크 (STO 비율별)
    for market in markets:
        market_scenarios = [sr for sr in sto_scenarios if sr['scenario']['market'] == market]
        market_scenarios = sorted(market_scenarios, key=lambda x: x['scenario']['sto_ratio'])
        
        ratios = [sr['scenario']['sto_ratio'] * 100 for sr in market_scenarios]
        retail_vars = [sr['metrics'].get('retail_VaR_95', 0) for sr in market_scenarios]
        
        axes[1, 0].plot(ratios, retail_vars, marker='o', linewidth=2, label=market)
    
    axes[1, 0].set_xlabel('STO 후순위 비율 (%)', fontsize=11)
    axes[1, 0].set_ylabel('개인 투자자 VaR 95% (억원)', fontsize=11)
    axes[1, 0].set_title('STO 비율별 개인 투자자 리스크', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 금융권 개선 효과
    for market in markets:
        baseline_var = baseline_by_market[market]['metrics']['VaR_95']
        
        market_scenarios = [sr for sr in sto_scenarios if sr['scenario']['market'] == market]
        market_scenarios = sorted(market_scenarios, key=lambda x: x['scenario']['sto_ratio'])
        
        ratios = [sr['scenario']['sto_ratio'] * 100 for sr in market_scenarios]
        improvements = [(baseline_var - sr['metrics']['VaR_95']) / baseline_var * 100 
                       for sr in market_scenarios]
        
        axes[1, 1].plot(ratios, improvements, marker='o', linewidth=2, label=market)
    
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('STO 후순위 비율 (%)', fontsize=11)
    axes[1, 1].set_ylabel('금융권 VaR 감소율 (%)', fontsize=11)
    axes[1, 1].set_title('STO 비율별 금융권 건전성 개선 효과', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 그래프 저장: {save_path}")
    
    plt.show()


def main():
    """메인 실행"""
    
    print("="*80)
    print("부동산 PF 시나리오 분석")
    print("="*80)
    print(f"\n총 {len(SCENARIOS)}개 시나리오 실행")
    print("  - STO 비율: 0% (기존), 15%, 28%, 40%")
    print("  - 시장 환경: 정상, 침체, 위기")
    
    # 축소 설정으로 빠른 테스트
    n_projects = 100
    n_simulations = 5000  # 빠른 실행을 위해 축소
    
    print(f"\n설정:")
    print(f"  프로젝트 수: {n_projects}개")
    print(f"  시뮬레이션 횟수: {n_simulations:,}회")
    
    input("\n계속하려면 Enter를 누르세요...")
    
    # 시나리오 실행
    scenario_results = []
    
    total_start = time.time()
    
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n진행: {i}/{len(SCENARIOS)}")
        
        result = run_scenario(scenario, n_projects, n_simulations)
        scenario_results.append(result)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"전체 시뮬레이션 완료!")
    print(f"총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"{'='*80}")
    
    # 결과 저장
    print("\n결과 정리 중...")
    
    # 1. 전체 비교표
    comparison_df = create_comparison_table(scenario_results)
    comparison_df.to_csv('scenario_comparison.csv', 
                         index=False, encoding='utf-8-sig')
    print("\n[전체 비교표]")
    print(comparison_df.to_string(index=False))
    print("\n  ✓ 저장: scenario_comparison.csv")
    
    # 2. 요약표 (기존 PF 대비)
    summary_df = create_summary_table(scenario_results)
    if summary_df is not None:
        summary_df.to_csv('scenario_summary.csv', 
                         index=False, encoding='utf-8-sig')
        print("\n[요약표 - 기존 PF 대비]")
        print(summary_df.to_string(index=False))
        print("\n  ✓ 저장: scenario_summary.csv")
    
    # 3. 시각화
    print("\n시각화 생성 중...")
    plot_scenario_comparison(
        scenario_results, 
        save_path='scenario_analysis.png'
    )
    
    # 4. 핵심 발견 요약
    print("\n" + "="*80)
    print("핵심 발견 요약")
    print("="*80)
    
    # 정상 시장에서 STO 비율별 비교
    normal_sto = [sr for sr in scenario_results 
                  if sr['scenario']['market'] == '정상' and sr['scenario']['sto_ratio'] > 0]
    baseline_normal = next(sr for sr in scenario_results 
                          if sr['scenario']['id'] == 0)
    
    print("\n1. 정상 시장에서 STO 비율별 효과:")
    for sr in sorted(normal_sto, key=lambda x: x['scenario']['sto_ratio']):
        sto_ratio = sr['scenario']['sto_ratio']
        fin_improve = (1 - sr['metrics']['VaR_95'] / baseline_normal['metrics']['VaR_95']) * 100
        retail_var = sr['metrics'].get('retail_VaR_95', 0)
        
        print(f"  STO {sto_ratio:.0%}: 금융권 {fin_improve:+.1f}%, 개인 VaR {retail_var:,.0f}억")
    
    # 침체/위기시 영향
    print("\n2. 시장 환경별 영향 (STO 28% 기준):")
    sto28 = [sr for sr in scenario_results if sr['scenario']['sto_ratio'] == 0.28]
    for sr in sto28:
        market = sr['scenario']['market']
        fin_var = sr['metrics']['VaR_95']
        retail_var = sr['metrics'].get('retail_VaR_95', 0)
        print(f"  {market}: 금융 {fin_var:,.0f}억, 개인 {retail_var:,.0f}억")
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    
    return scenario_results


if __name__ == "__main__":
    results = main()