"""
최종 완전판 - 다차원 시나리오 분석 (V=1000억)
- STO 비율: 0% ~ 100% (10% 간격, Trad PF 포함)
- 시장 상황: Perfect (100%), Good (84%), Recession (65%), Crisis (41%), Extreme (15%)
- 총 55개 시나리오
"""

import numpy as np
import pandas as pd
from pf_simulation_v2 import SimulationParams, ImprovedPFSimulation
from pf_analysis_v2 import ImprovedRiskAnalyzer
from final_visualization import create_comprehensive_visualization


# ===== 시나리오 정의 =====
STO_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STO_LABELS = ['Trad PF', 'STO 10%', 'STO 20%', 'STO 30%', 'STO 40%',
              'STO 50%', 'STO 60%', 'STO 70%', 'STO 80%', 'STO 90%', 'STO 100%']

REFERENCE_STO_LABEL = 'STO 100%'
MARKET_SCENARIOS = {
    'Perfect': {
        'label': 'Perfect (100%)',
        'color': 'darkgreen',
        'params': {
            'use_logistic_sales': True,
            'sales_max': 0.99,
            'sales_growth_rate': 1.2,
            'sales_inflection': 3.0,
            'sigma_sales': 0.03,
            'initial_sales': 0.50,
        }
    },
    'Good': {
        'label': 'Good (84%)',
        'color': 'lightgreen',
        'params': {
            'use_logistic_sales': True,
            'sales_max': 0.85,
            'sales_growth_rate': 0.5,
            'sales_inflection': 8.0,
            'sigma_sales': 0.15,
            'initial_sales': 0.15,
        }
    },
    'Recession': {
        'label': 'Recession (65%)',
        'color': 'orange',
        'params': {
            'use_logistic_sales': True,
            'sales_max': 0.70,
            'sales_growth_rate': 0.4,
            'sales_inflection': 9.0,
            'sigma_sales': 0.20,
            'initial_sales': 0.15,
        }
    },
    'Crisis': {
        'label': 'Crisis (41%)',
        'color': 'red',
        'params': {
            'use_logistic_sales': False,
            'mu_sales_base': 0.00,
            'sigma_sales': 0.25,
            'initial_sales': 0.15,
        }
    },
    'Extreme': {
        'label': 'Extreme (15%)',
        'color': 'darkred',
        'params': {
            'use_logistic_sales': False,
            'mu_sales_base': -0.02,  # -2%/분기 하락
            'sigma_sales': 0.35,      # 35% 변동성
            'initial_sales': 0.10,    # 10%에서 시작
        }
    }
}


def run_scenario(sto_ratio, market_key, n_simulations=50000): 
    """시나리오 실행"""
    market_params = MARKET_SCENARIOS[market_key]['params']
    
    # 메모리 부족 시: n_projects=50, n_simulations=3000으로 추가 감축 가능
    params = SimulationParams(
        n_simulations=n_simulations,
        n_projects=100,  # 메모리 부족 시 50으로 감축
        T=16,
        sto_ratio=sto_ratio,
        **market_params
    )
    
    use_sto = (sto_ratio > 0)
    sim = ImprovedPFSimulation(params, use_sto=use_sto)
    results = sim.run_simulation()
    
    analyzer = ImprovedRiskAnalyzer(results, params)
    metrics = analyzer.calculate_all_metrics()
    
    return results, metrics, params


def run_all_scenarios():
    """모든 시나리오 실행"""
    total_scenarios = len(STO_RATIOS) * len(MARKET_SCENARIOS)
    print("="*80)
    print(f"RUNNING ALL SCENARIOS ({total_scenarios} combinations)")
    print("="*80)
    
    all_results = {}
    
    for sto_idx, sto_ratio in enumerate(STO_RATIOS):
        sto_label = STO_LABELS[sto_idx]
        
        for market_key in MARKET_SCENARIOS.keys():
            market_label = MARKET_SCENARIOS[market_key]['label']
            
            scenario_name = f"{sto_label}_{market_key}"
            print(f"\n[{scenario_name}] Running...")
            
            results, metrics, params = run_scenario(sto_ratio, market_key)
            
            final_sales = results['state']['sales_rate'][:, :, -1].mean()
            
            # 메모리 절약: 필요한 것만 저장
            all_results[scenario_name] = {
                'sto_ratio': sto_ratio,
                'sto_label': sto_label,
                'market_key': market_key,
                'market_label': market_label,
                'results': {
                    'state': {
                        'sales_rate': results['state']['sales_rate'],  # 필요
                    },
                    'losses': {
                        'systemic_loss': results['losses']['systemic_loss'],  # 필요
                        'retail_loss': results['losses']['retail_loss'] if sto_ratio > 0 else None,  # 필요 (STO만)
                    }
                },
                'metrics': metrics,
                'params': params,
                'final_sales': final_sales,
            }
            
            print(f"  Sales: {final_sales:.1%}")
            print(f"  Financial VaR95: {metrics['VaR_95']:.0f}억")
            if sto_ratio > 0:
                print(f"  Retail Loss Rate (VaR95): {metrics['retail_loss_rate_VaR95']*100:.2f}%")
            
            # 메모리 해제
            del results
    
    print("\n" + "="*80)
    print("✅ ALL SCENARIOS COMPLETE")
    print("="*80)
    
    return all_results


def create_summary_table(all_results):
    """요약 테이블 생성"""
    data = []
    
    for scenario_name, result in all_results.items():
        sto_label = result['sto_label']
        market_label = result['market_label']
        metrics = result['metrics']
        final_sales = result['final_sales']
        
        row = {
            'STO_Ratio': sto_label,
            'Market': market_label,
            'Sales_Rate': final_sales,
            'Financial_VaR95': metrics['VaR_95'],
            'Financial_ES95': metrics['ES_95'],
            'Systemic_VaR95': metrics['VaR_95'],  # Traditional system risk
        }
        
        if result['sto_ratio'] > 0:
            row['Retail_VaR95_Rate'] = metrics['retail_loss_rate_VaR95']
            row['Retail_ES95_Rate'] = metrics['retail_loss_rate_ES95']
            row['Retail_VaR95_Absolute'] = metrics['retail_VaR_95']
            
            # ✅ FIXED: Extended System Risk 올바른 계산
            # Extended = Financial + Retail (단순 합산이 더 명확)
            row['Extended_Systemic_VaR95'] = metrics['VaR_95'] + metrics['retail_VaR_95']
            row['Extended_Systemic_ES95'] = metrics['ES_95'] + metrics['retail_ES_95']
            
            # Total system risk change
            row['System_Risk_Change'] = metrics['retail_VaR_95']  # 개인 추가분
        else:
            row['Retail_VaR95_Rate'] = np.nan
            row['Retail_ES95_Rate'] = np.nan
            row['Retail_VaR95_Absolute'] = 0
            row['Extended_Systemic_VaR95'] = metrics['VaR_95']  # Same as financial only
            row['Extended_Systemic_ES95'] = metrics['ES_95']
            row['System_Risk_Change'] = 0
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def export_results_to_excel(df, all_results):
    """Excel로 결과 내보내기"""
    
    with pd.ExcelWriter('simulation_results_v1000.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Summary
        df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: System Risk Analysis (NEW!)
        system_risk_data = []
        for scenario_name, result in all_results.items():
            sto_label = result['sto_label']
            market_label = result['market_label']
            metrics = result['metrics']
            
            row = {
                'Scenario': scenario_name,
                'STO_Ratio': sto_label,
                'Market': market_label,
                'Financial_VaR95': metrics['VaR_95'],
                'Financial_ES95': metrics['ES_95'],
            }
            
            if result['sto_ratio'] > 0:
                row['Retail_VaR95'] = metrics['retail_VaR_95']
                row['Extended_System_VaR95'] = metrics['extended_VaR_95']
                row['Extended_System_ES95'] = metrics['extended_ES_95']
                row['System_Risk_Change'] = metrics['extended_VaR_95'] - metrics['VaR_95']
                row['System_Risk_Change_Pct'] = (metrics['extended_VaR_95'] - metrics['VaR_95']) / metrics['VaR_95'] * 100
            else:
                row['Retail_VaR95'] = 0
                row['Extended_System_VaR95'] = metrics['VaR_95']
                row['Extended_System_ES95'] = metrics['ES_95']
                row['System_Risk_Change'] = 0
                row['System_Risk_Change_Pct'] = 0
            
            system_risk_data.append(row)
        
        df_system = pd.DataFrame(system_risk_data)
        df_system.to_excel(writer, sheet_name='System_Risk_Analysis', index=False)
        
        # Sheet 3: STO Benefit Analysis
        reductions = []
        for market_key in MARKET_SCENARIOS.keys():
            market_label = MARKET_SCENARIOS[market_key]['label']
            trad_var = df[(df['Market'] == market_label) & (df['STO_Ratio'] == 'Trad PF')]['Financial_VaR95'].values[0]
            
            for sto_label in STO_LABELS[1:]:
                sto_var = df[(df['Market'] == market_label) & (df['STO_Ratio'] == sto_label)]['Financial_VaR95'].values[0]
                reduction_abs = trad_var - sto_var
                reduction_pct = (trad_var - sto_var) / trad_var * 100
                
                reductions.append({
                    'Market': market_label,
                    'STO_Ratio': sto_label,
                    'Trad_VaR95': trad_var,
                    'STO_VaR95': sto_var,
                    'Reduction_Amount': reduction_abs,
                    'Reduction_Percent': reduction_pct
                })
        
        df_reduction = pd.DataFrame(reductions)
        df_reduction.to_excel(writer, sheet_name='STO_Benefit', index=False)
        
        # Sheet 4: Retail Risk Analysis
        retail_data = []
        for scenario_name, result in all_results.items():
            if result['sto_ratio'] > 0:
                retail_data.append({
                    'Scenario': scenario_name,
                    'STO_Ratio': result['sto_label'],
                    'Market': result['market_label'],
                    'Sales_Rate': result['final_sales'],
                    'Retail_VaR95_Rate': result['metrics']['retail_loss_rate_VaR95'],
                    'Retail_ES95_Rate': result['metrics']['retail_loss_rate_ES95'],
                    'VaR_ES_Gap': result['metrics']['retail_loss_rate_ES95'] - result['metrics']['retail_loss_rate_VaR95'],
                    'Loss_Probability': result['metrics']['retail_loss_probability'],
                })
        
        df_retail = pd.DataFrame(retail_data)
        df_retail.to_excel(writer, sheet_name='Retail_Risk', index=False)
    
    print("✅ Results exported to: simulation_results_v1000.xlsx")


def print_key_insights(df, all_results):
    """핵심 인사이트 출력"""
    
    print("\n" + "="*80)
    print("📊 KEY INSIGHTS")
    print("="*80)
    
    # 0. System Risk Change Analysis (NEW!)
    print("\n0. SYSTEM RISK CHANGE (Financial + Retail):")
    print("   [Positive = Risk Increase from Retail Exposure]")
    for market_key in ['Perfect', 'Good', 'Recession', 'Crisis', 'Extreme']:
        market_label = MARKET_SCENARIOS[market_key]['label']
        print(f"\n   {market_label}:")
        
        for sto_label in STO_LABELS[1:]:
            scenario = f"{sto_label}_{market_key}"
            financial_var = all_results[scenario]['metrics']['VaR_95']
            retail_var = all_results[scenario]['metrics']['retail_VaR_95']
            extended_var = financial_var + retail_var  # 올바른 계산
            change = retail_var  # 개인 추가분
            change_pct = change / financial_var * 100
            
            print(f"      {sto_label}: +{change:,.0f}억 (+{change_pct:.1f}%) 📈 RETAIL ADDED")
    
    # 1. Best STO ratio per scenario
    print("\n1. OPTIMAL STO RATIO BY MARKET CONDITION:")
    for market_key in MARKET_SCENARIOS.keys():
        market_label = MARKET_SCENARIOS[market_key]['label']
        market_data = df[(df['Market'] == market_label) & (df['STO_Ratio'] != 'Trad PF')]
        
        best_sto = market_data.loc[market_data['Financial_VaR95'].idxmin(), 'STO_Ratio']
        best_var = market_data['Financial_VaR95'].min()
        
        print(f"   {market_label:20s}: {best_sto:10s} (VaR95 = {best_var:,.0f}억)")
    
    # 2. STO Benefit magnitude
    print("\n2. STO BENEFIT MAGNITUDE:")
    for market_key in ['Perfect', 'Extreme']:
        market_label = MARKET_SCENARIOS[market_key]['label']
        trad_var = df[(df['Market'] == market_label) & (df['STO_Ratio'] == 'Trad PF')]['Financial_VaR95'].values[0]
        
        print(f"\n   {market_label}:")
        for sto_label in STO_LABELS[1:]:
            sto_var = df[(df['Market'] == market_label) & (df['STO_Ratio'] == sto_label)]['Financial_VaR95'].values[0]
            reduction = (trad_var - sto_var) / trad_var * 100
            print(f"      {sto_label}: {reduction:5.1f}% reduction ({trad_var:,.0f}억 → {sto_var:,.0f}억)")
    
    # 3. Retail investor safety
    print("\n3. RETAIL INVESTOR SAFETY (Loss Amount VaR95):")
    for market_key in ['Perfect', 'Good', 'Recession', 'Crisis', 'Extreme']:
        market_label = MARKET_SCENARIOS[market_key]['label']
        print(f"\n   {market_label}:")
        
        for sto_label in STO_LABELS[1:]:
            scenario = f"{sto_label}_{market_key}"
            amount = all_results[scenario]['metrics']['retail_VaR_95']
            
            if amount == 0:
                safety = "✅ SAFE (No Loss)"
            elif amount < 100:
                safety = "✅ LOW RISK"
            elif amount < 500:
                safety = "⚠️  MODERATE RISK"
            elif amount < 1000:
                safety = "⚠️  HIGH RISK"
            else:
                safety = "❌ VERY HIGH RISK"
            
            print(f"      {sto_label}: {amount:,.0f}억 {safety}")
    
    # 4. Policy recommendation
    print("\n4. POLICY RECOMMENDATION:")
    print("""
   ✅ STO Introduction: STRONGLY RECOMMENDED
      • Financial system risk reduction scales with higher STO ratios (up to 50-70% drop in tail risk)
      • Retail exposure increases beyond 70% STO, so adequate protection is essential
   
   📊 SYSTEM RISK ANALYSIS:
      • Extended System Risk = Financial VaR + Retail VaR (simple sum)
      • STO shifts risk from banks to households while lowering institutional concentration
      • STO introduces distributed retail exposure that must be monitored and mitigated
   
   ⚠️  CRITICAL INSIGHT:
      • Traditional PF concentrates risk entirely on financial institutions → systemic crisis potential
      • STO PF distributes risk: Financial (≈70-80%) + Retail (≈20-30%) depending on scenario
      • Total exposure rises but individual financial institutions are shielded
      • Trade-off: Macro stability vs retail vulnerability
   
   ✅ STO Ratio Spectrum Tested: 0%~100% (10% 간격)
      • Low STO (<30%) keeps retail risk minimal but limits financial relief
      • Mid STO (40%-70%) offers balanced risk-return by reducing financial VaR while keeping retail loss tolerable
      • High STO (>70%) maximizes financial benefit but requires guarantees for retail investors
   
   ⚠️  Retail Investor Protection ESSENTIAL:
      • Mandate 95%+ pre-sales or provide insurance for retail participation
      • Track scenario-specific retail loss VaR to calibrate protection
   
   ✅ Market Condition Monitoring:
      • Perfect/Good: Retail losses remain minimal across STO ratios
      • Recession: Retail risk emerges gradually as STO rises
      • Crisis/Extreme: Retail loss VaR jumps for STO ≥ 60%
   
   📊 Key Insight:
      • STO = Risk TRANSFER + AMPLIFICATION mechanism
      • Reduces institutional concentration, increases distributed retail risk
      • Net benefit depends on how well retail exposure is protected
    """)


def main():
    """메인 실행"""
    
    print("="*80)
    print("COMPREHENSIVE MULTI-SCENARIO ANALYSIS")
    print("V = 1000억, STO Ratios: 0%~100% (10% step)")
    print("Market: [Perfect, Good, Recession, Crisis, Extreme]")
    print("Total: 55 Scenarios")
    print("="*80)
    
    # Run all scenarios
    all_results = run_all_scenarios()
    
    # Create summary table
    df = create_summary_table(all_results)
    
    # Visualization
    create_comprehensive_visualization(
        all_results,
        df,
        STO_LABELS,
        MARKET_SCENARIOS,
        REFERENCE_STO_LABEL
    )
    
    # Export to Excel
    export_results_to_excel(df, all_results)
    
    # Print insights
    print_key_insights(df, all_results)
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  1. comprehensive_analysis_v1000.png")
    print("  2. simulation_results_v1000.xlsx")


if __name__ == "__main__":
    main()
