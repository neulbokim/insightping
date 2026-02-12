"""
개선된 부동산 PF 몬테카를로 시뮬레이션 실행 스크립트
"""

import numpy as np
import pandas as pd
import time
from insightping.js_simulation.pf_simulation_v2 import SimulationParams, ImprovedPFSimulation
from pf_analysis_v2 import ImprovedRiskAnalyzer, ImprovedVisualizer

def main():
    print("="*80)
    print("개선된 부동산 PF 몬테카를로 시뮬레이션")
    print("="*80)
    
    # 파라미터 설정
    params = SimulationParams(
        n_projects=100,
        T=16,  # 4년
        n_simulations=10000,
        mu_sales_base=0.06,
        sigma_sales=0.12,
        sto_ratio=0.28,
    )
    
    print("\n[파라미터 설정]")
    print(f"  프로젝트 수: {params.n_projects}개")
    print(f"  시뮬레이션 기간: {params.T}분기 (4년)")
    print(f"  몬테카를로 시행: {params.n_simulations:,}회")
    print(f"  STO 후순위 비율: {params.sto_ratio:.1%}")
    print(f"  기본 분양률 증가: {params.mu_sales_base:.1%}/분기")
    print(f"  분양률 변동성: {params.sigma_sales:.1%}")
    
    # ========== 1. 기존 PF ==========
    print("\n" + "="*80)
    print("1. 기존 부동산 PF 시뮬레이션")
    print("="*80)
    
    start = time.time()
    sim_trad = ImprovedPFSimulation(params, use_sto=False)
    results_trad = sim_trad.run_simulation()
    elapsed_trad = time.time() - start
    
    print(f"✓ 완료 (소요: {elapsed_trad:.1f}초)")
    
    analyzer_trad = ImprovedRiskAnalyzer(results_trad, params)
    metrics_trad = analyzer_trad.calculate_all_metrics()
    analyzer_trad.print_summary()
    
    # ========== 2. STO PF ==========
    print("\n" + "="*80)
    print("2. STO 도입 부동산 PF 시뮬레이션")
    print("="*80)
    
    start = time.time()
    sim_sto = ImprovedPFSimulation(params, use_sto=True)
    results_sto = sim_sto.run_simulation()
    elapsed_sto = time.time() - start
    
    print(f"✓ 완료 (소요: {elapsed_sto:.1f}초)")
    
    analyzer_sto = ImprovedRiskAnalyzer(results_sto, params)
    metrics_sto = analyzer_sto.calculate_all_metrics()
    analyzer_sto.print_summary()
    
    # ========== 3. 비교 분석 ==========
    print("\n" + "="*80)
    print("3. 기존 PF vs STO PF 비교")
    print("="*80)
    
    comparison = pd.DataFrame({
        '지표': [
            'VaR 95% (금융권)',
            'ES 95% (금융권)',
            'VaR 99% (금융권)',
            'ES 99% (금융권)',
            '10% 이상 동시 부실 확률',
            '20% 이상 동시 부실 확률',
            '30% 이상 동시 부실 확률',
            '평균 전이 시간 (분기)',
            '전이 발생 확률',
            '전체 회수율',
        ],
        '기존 PF': [
            f"{metrics_trad['VaR_95']:,.0f}",
            f"{metrics_trad['ES_95']:,.0f}",
            f"{metrics_trad['VaR_99']:,.0f}",
            f"{metrics_trad['ES_99']:,.0f}",
            f"{metrics_trad['prob_10pct_simultaneous']:.2%}",
            f"{metrics_trad['prob_20pct_simultaneous']:.2%}",
            f"{metrics_trad['prob_30pct_simultaneous']:.2%}",
            f"{metrics_trad['mean_contagion_time']:.2f}",
            f"{metrics_trad['prob_contagion']:.2%}",
            f"{metrics_trad['overall_recovery_rate']:.2%}",
        ],
        'STO PF': [
            f"{metrics_sto['VaR_95']:,.0f}",
            f"{metrics_sto['ES_95']:,.0f}",
            f"{metrics_sto['VaR_99']:,.0f}",
            f"{metrics_sto['ES_99']:,.0f}",
            f"{metrics_sto['prob_10pct_simultaneous']:.2%}",
            f"{metrics_sto['prob_20pct_simultaneous']:.2%}",
            f"{metrics_sto['prob_30pct_simultaneous']:.2%}",
            f"{metrics_sto['mean_contagion_time']:.2f}",
            f"{metrics_sto['prob_contagion']:.2%}",
            f"{metrics_sto['overall_recovery_rate']:.2%}",
        ],
        '변화': [
            f"{(metrics_sto['VaR_95'] / metrics_trad['VaR_95'] - 1) * 100:+.1f}%",
            f"{(metrics_sto['ES_95'] / metrics_trad['ES_95'] - 1) * 100:+.1f}%",
            f"{(metrics_sto['VaR_99'] / metrics_trad['VaR_99'] - 1) * 100:+.1f}%",
            f"{(metrics_sto['ES_99'] / metrics_trad['ES_99'] - 1) * 100:+.1f}%",
            f"{(metrics_sto['prob_10pct_simultaneous'] - metrics_trad['prob_10pct_simultaneous']) * 100:+.2f}%p",
            f"{(metrics_sto['prob_20pct_simultaneous'] - metrics_trad['prob_20pct_simultaneous']) * 100:+.2f}%p",
            f"{(metrics_sto['prob_30pct_simultaneous'] - metrics_trad['prob_30pct_simultaneous']) * 100:+.2f}%p",
            f"{metrics_sto['mean_contagion_time'] - metrics_trad['mean_contagion_time']:+.2f}",
            f"{(metrics_sto['prob_contagion'] - metrics_trad['prob_contagion']) * 100:+.2f}%p",
            f"{(metrics_sto['overall_recovery_rate'] - metrics_trad['overall_recovery_rate']) * 100:+.2f}%p",
        ]
    })
    
    print("\n", comparison.to_string(index=False))
    
    # 개인 투자자 손실
    print("\n" + "-"*80)
    print("STO 개인 투자자 손실")
    print("-"*80)
    print(f"  VaR 95%: {metrics_sto['retail_VaR_95']:,.0f} 억원")
    print(f"  ES 95%:  {metrics_sto['retail_ES_95']:,.0f} 억원")
    print(f"  VaR 99%: {metrics_sto['retail_VaR_99']:,.0f} 억원")
    print(f"  ES 99%:  {metrics_sto['retail_ES_99']:,.0f} 억원")
    print(f"  평균 손실률: {metrics_sto['retail_loss_rate']:.2%}")
    
    # 확장 시스템
    print("\n" + "-"*80)
    print("STO 확장 시스템 손실 (금융 + 가계)")
    print("-"*80)
    print(f"  VaR 95%: {metrics_sto['extended_VaR_95']:,.0f} 억원")
    print(f"  ES 95%:  {metrics_sto['extended_ES_95']:,.0f} 억원")
    print(f"  기존 대비: {(metrics_sto['extended_VaR_95'] / metrics_trad['VaR_95'] - 1) * 100:+.1f}%")
    
    # ========== 4. 시각화 ==========
    print("\n" + "="*80)
    print("4. 시각화")
    print("="*80)
    
    viz = ImprovedVisualizer(results_trad, results_sto, params)
    
    print("\n  [1] 손실 분포...")
    viz.plot_loss_distribution()
    
    print("  [2] 시간별 진화...")
    viz.plot_time_evolution()
    
    print("  [3] 리스크 지표 비교...")
    viz.plot_metrics_comparison(analyzer_trad, analyzer_sto)
    
    # ========== 5. 결과 저장 ==========
    print("\n" + "="*80)
    print("5. 결과 저장")
    print("="*80)
    
    comparison.to_csv('/mnt/user-data/outputs/pf_comparison_v2.csv', 
                     index=False, encoding='utf-8-sig')
    print("  ✓ 비교표: pf_comparison_v2.csv")
    
    # 상세 지표
    detailed = pd.DataFrame([
        {'구분': '기존 PF', **{k: v for k, v in metrics_trad.items() if isinstance(v, (int, float))}},
        {'구분': 'STO PF', **{k: v for k, v in metrics_sto.items() if isinstance(v, (int, float))}}
    ])
    detailed.to_csv('/mnt/user-data/outputs/pf_detailed_v2.csv', 
                   index=False, encoding='utf-8-sig')
    print("  ✓ 상세 지표: pf_detailed_v2.csv")
    
    # 요약 보고서
    with open('/mnt/user-data/outputs/pf_summary_v2.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("부동산 PF 몬테카를로 시뮬레이션 결과 요약\n")
        f.write("="*80 + "\n\n")
        
        f.write("[핵심 발견]\n\n")
        
        f.write("1. 금융권 건전성\n")
        var_change = (metrics_sto['VaR_95'] / metrics_trad['VaR_95'] - 1) * 100
        f.write(f"   - STO 도입시 금융권 VaR 95% 변화: {var_change:+.1f}%\n")
        f.write(f"   - 기존 PF: {metrics_trad['VaR_95']:,.0f}억원\n")
        f.write(f"   - STO PF: {metrics_sto['VaR_95']:,.0f}억원\n")
        if var_change < 0:
            f.write("   → 금융권 Tail Risk 감소 (긍정적)\n\n")
        else:
            f.write("   → 금융권 Tail Risk 증가 (부정적)\n\n")
        
        f.write("2. 개인 투자자 보호\n")
        f.write(f"   - 개인 투자자 VaR 95%: {metrics_sto['retail_VaR_95']:,.0f}억원\n")
        f.write(f"   - 개인 투자자 평균 손실률: {metrics_sto['retail_loss_rate']:.2%}\n")
        f.write("   → 후순위 투자자는 높은 손실 위험 직면 (우려)\n\n")
        
        f.write("3. 시스템 리스크\n")
        ext_change = (metrics_sto['extended_VaR_95'] / metrics_trad['VaR_95'] - 1) * 100
        f.write(f"   - 확장 시스템 손실 (금융+가계): {metrics_sto['extended_VaR_95']:,.0f}억원\n")
        f.write(f"   - 기존 대비 변화: {ext_change:+.1f}%\n")
        if ext_change > 0:
            f.write("   → 전체 시스템 리스크 증가 (우려)\n\n")
        else:
            f.write("   → 전체 시스템 리스크 감소 (긍정적)\n\n")
        
        f.write("4. 정책 시사점\n")
        f.write("   - 금융권 건전성과 개인 투자자 보호 간 트레이드오프 존재\n")
        f.write("   - 후순위 비율 제한, 개인 투자 한도 설정 필요\n")
        f.write("   - 정보 공시 강화 및 투자자 교육 필수\n")
        f.write("   - 소비 위축 피드백 효과 모니터링 필요\n")
    
    print("  ✓ 요약 보고서: pf_summary_v2.txt")
    
    print("\n" + "="*80)
    print("시뮬레이션 완료!")
    print("="*80)
    
    return results_trad, results_sto, metrics_trad, metrics_sto


if __name__ == "__main__":
    results_trad, results_sto, metrics_trad, metrics_sto = main()
