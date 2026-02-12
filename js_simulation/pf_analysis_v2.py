"""
개선된 부동산 PF 시뮬레이션 분석 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class ImprovedRiskAnalyzer:
    """개선된 시스템 리스크 분석"""
    
    def __init__(self, results: Dict, params):
        self.results = results
        self.params = params
        self.metrics = {}
        
    def calculate_var_es(self, loss_distribution: np.ndarray,
                        confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """VaR 및 ES 계산"""
        metrics = {}
        
        # 평균 추가
        metrics['mean'] = loss_distribution.mean()
        metrics['median'] = np.median(loss_distribution)
        
        for alpha in confidence_levels:
            var = np.percentile(loss_distribution, alpha * 100)
            metrics[f'VaR_{int(alpha*100)}'] = var
            
            tail_losses = loss_distribution[loss_distribution > var]
            es = tail_losses.mean() if len(tail_losses) > 0 else var
            metrics[f'ES_{int(alpha*100)}'] = es
        
        return metrics
    
    def calculate_default_probability(self) -> Dict[str, float]:
        """동시 부실 발생 확률"""
        state = self.results['state']
        p = self.params
        
        n_defaults = (~state['refinance_success']).sum(axis=1)
        mean_defaults = n_defaults.mean(axis=0)
        max_defaults = n_defaults.max(axis=1)
        
        return {
            'mean_defaults_final': mean_defaults[-1],
            'prob_10pct_simultaneous': (max_defaults >= p.n_projects * 0.1).mean(),
            'prob_20pct_simultaneous': (max_defaults >= p.n_projects * 0.2).mean(),
            'prob_30pct_simultaneous': (max_defaults >= p.n_projects * 0.3).mean(),
            'max_defaults_mean': max_defaults.mean(),
            'max_defaults_95th': np.percentile(max_defaults, 95),
            'max_defaults_99th': np.percentile(max_defaults, 99),
        }
    
    def calculate_contagion_speed(self) -> Dict[str, float]:
        """전이 속도 측정"""
        losses = self.results['losses']
        p = self.params
        
        systemic_loss = losses['systemic_loss']
        threshold = p.n_projects * p.total_project_value * p.systemic_threshold
        
        contagion_time = np.zeros(p.n_simulations)
        
        for i in range(p.n_simulations):
            exceed_idx = np.where(systemic_loss[i, :] > threshold)[0]
            contagion_time[i] = exceed_idx[0] if len(exceed_idx) > 0 else p.T
        
        return {
            'mean_contagion_time': contagion_time.mean(),
            'median_contagion_time': np.median(contagion_time),
            'prob_contagion': (contagion_time < p.T).mean(),
            'contagion_speed_5th': np.percentile(contagion_time, 5),
            'contagion_speed_95th': np.percentile(contagion_time, 95),
        }
    
    def calculate_recovery_rate_stats(self) -> Dict[str, float]:
        """회수율 통계"""
        state = self.results['state']
        losses = self.results['losses']
        p = self.params
        
        # 총 청구액
        total_claim = state['securitization_balance'][:, :, 0].sum()
        
        # 총 손실
        total_sec_loss = losses['securities_loss'].sum()
        
        # 회수율 = 1 - (손실 / 청구액)
        recovery_rate = 1 - (total_sec_loss / (total_claim + 1e-10))
        
        return {
            'overall_recovery_rate': recovery_rate,
        }
    
    def calculate_all_metrics(self) -> Dict:
        """모든 리스크 지표 계산"""
        losses = self.results['losses']
        
        # 최종 시점 시스템 손실
        final_systemic = losses['systemic_loss'][:, -1]
        
        # 누적 시스템 손실
        cumulative_systemic = losses['systemic_loss'].sum(axis=1)
        
        # VaR & ES
        var_es = self.calculate_var_es(cumulative_systemic)
        
        # 동시 부실
        default_prob = self.calculate_default_probability()
        
        # 전이 속도
        contagion = self.calculate_contagion_speed()
        
        # 회수율
        recovery = self.calculate_recovery_rate_stats()
        
        self.metrics = {
            **var_es,
            **default_prob,
            **contagion,
            **recovery,
        }
        
        # STO 추가 지표
        if 'retail_loss' in losses:
            # 개인 투자자 손실 (최종 시점 기준)
            retail_total = losses['retail_loss'][:, :, -1].sum(axis=1)  # (n_sim,) 시뮬레이션별 총 손실
            retail_metrics = self.calculate_var_es(retail_total)
            retail_metrics = {f'retail_{k}': v for k, v in retail_metrics.items()}
            self.metrics.update(retail_metrics)
            
            # 확장 시스템 손실 (최종 시점 기준)
            extended_total = losses['systemic_loss_extended'][:, -1]  # 마지막 시점만!
            extended_metrics = self.calculate_var_es(extended_total)
            extended_metrics = {f'extended_{k}': v for k, v in extended_metrics.items()}
            self.metrics.update(extended_metrics)
            
            # 개인 투자자 초기 투자금
            initial_junior = (
                self.params.n_projects * 
                self.params.total_project_value * 
                self.params.securitization_ratio * 
                self.params.sto_ratio
            )
            self.metrics['retail_initial_investment'] = initial_junior
            
            # 개인 투자자 손실률 통계 (NEW!)
            loss_rates = retail_total / (initial_junior + 1e-10)
            self.metrics['retail_loss_rate_mean'] = loss_rates.mean()
            self.metrics['retail_loss_rate_median'] = np.median(loss_rates)
            self.metrics['retail_loss_rate_VaR95'] = np.percentile(loss_rates, 95)
            self.metrics['retail_loss_rate_VaR99'] = np.percentile(loss_rates, 99)
            
            # ES Rate (Expected Shortfall Rate)
            VaR95_threshold = np.percentile(loss_rates, 95)
            tail_losses_95 = loss_rates[loss_rates >= VaR95_threshold]
            if len(tail_losses_95) > 0:
                self.metrics['retail_loss_rate_ES95'] = tail_losses_95.mean()
            else:
                self.metrics['retail_loss_rate_ES95'] = 0.0
            
            VaR99_threshold = np.percentile(loss_rates, 99)
            tail_losses_99 = loss_rates[loss_rates >= VaR99_threshold]
            if len(tail_losses_99) > 0:
                self.metrics['retail_loss_rate_ES99'] = tail_losses_99.mean()
            else:
                self.metrics['retail_loss_rate_ES99'] = 0.0
            
            # 손실 발생 확률 (손실 > 0인 시뮬레이션 비율)
            loss_occurred_rate = (retail_total > 0).mean()
            self.metrics['retail_loss_probability'] = loss_occurred_rate
            
            # 손실 발생 시 평균 손실 (조건부)
            if loss_occurred_rate > 0:
                conditional_loss = retail_total[retail_total > 0].mean()
                self.metrics['retail_conditional_loss'] = conditional_loss
            else:
                self.metrics['retail_conditional_loss'] = 0.0
        
        return self.metrics
    
    def print_summary(self):
        """지표 요약 출력"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*70)
        print("시스템 리스크 지표 요약")
        print("="*70)
        
        print("\n1. Tail Risk (VaR & ES) - 누적 손실")
        print(f"  VaR 95%: {self.metrics['VaR_95']:,.0f} 억원")
        print(f"  ES 95%:  {self.metrics['ES_95']:,.0f} 억원")
        print(f"  VaR 99%: {self.metrics['VaR_99']:,.0f} 억원")
        print(f"  ES 99%:  {self.metrics['ES_99']:,.0f} 억원")
        
        print("\n2. 동시 부실 확률")
        print(f"  평균 부실 프로젝트 (최종 시점): {self.metrics['mean_defaults_final']:.1f}개")
        print(f"  10개 이상 동시 부실 발생 확률: {self.metrics['prob_10pct_simultaneous']:.1%}")
        print(f"  20개 이상 동시 부실 발생 확률: {self.metrics['prob_20pct_simultaneous']:.1%}")
        print(f"  30개 이상 동시 부실 발생 확률: {self.metrics['prob_30pct_simultaneous']:.1%}")
        print(f"  전체 기간 중 최대 동시 부실 평균: {self.metrics['max_defaults_mean']:.1f}개")
        print(f"  전체 기간 중 최대 동시 부실 95th: {self.metrics['max_defaults_95th']:.1f}개")
        
        print("\n3. 전이 속도")
        print(f"  평균 전이 시간: {self.metrics['mean_contagion_time']:.1f} 분기")
        print(f"  중앙값 전이 시간: {self.metrics['median_contagion_time']:.1f} 분기")
        print(f"  전이 발생 확률: {self.metrics['prob_contagion']:.1%}")
        print(f"  빠른 전이 (5th): {self.metrics['contagion_speed_5th']:.1f} 분기")
        
        print("\n4. 회수율")
        print(f"  전체 회수율: {self.metrics['overall_recovery_rate']:.1%}")
        
        if 'retail_VaR_95' in self.metrics:
            print("\n5. 개인 투자자 손실 (STO)")
            print(f"  VaR 95%: {self.metrics['retail_VaR_95']:,.0f} 억원")
            print(f"  ES 95%:  {self.metrics['retail_ES_95']:,.0f} 억원")
            print(f"  VaR 99%: {self.metrics['retail_VaR_99']:,.0f} 억원")
            print(f"  ES 99%:  {self.metrics['retail_ES_99']:,.0f} 억원")
            print(f"  평균 손실률: {self.metrics['retail_loss_rate']:.2%}")
            
            print("\n6. 확장 시스템 손실 (금융 + 가계)")
            print(f"  VaR 95%: {self.metrics['extended_VaR_95']:,.0f} 억원")
            print(f"  ES 95%:  {self.metrics['extended_ES_95']:,.0f} 억원")


class ImprovedVisualizer:
    """개선된 시각화"""
    
    def __init__(self, results_trad: Dict, results_sto: Dict, params):
        self.results_trad = results_trad
        self.results_sto = results_sto
        self.params = params
        
    def plot_loss_distribution(self, save_path: str = None):
        """손실 분포 비교"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 누적 손실
        loss_trad = self.results_trad['losses']['systemic_loss'].sum(axis=1)
        loss_sto_fin = self.results_sto['losses']['systemic_loss'].sum(axis=1)
        loss_sto_ext = self.results_sto['losses']['systemic_loss_extended'].sum(axis=1)
        
        # 히스토그램
        axes[0, 0].hist(loss_trad, bins=50, alpha=0.6, label='기존 PF', density=True, color='blue')
        axes[0, 0].hist(loss_sto_fin, bins=50, alpha=0.6, label='STO PF (금융)', density=True, color='orange')
        axes[0, 0].set_xlabel('누적 시스템 손실 (억원)', fontsize=11)
        axes[0, 0].set_ylabel('확률 밀도', fontsize=11)
        axes[0, 0].set_title('금융권 손실 분포', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # CDF
        sorted_trad = np.sort(loss_trad)
        sorted_sto_fin = np.sort(loss_sto_fin)
        sorted_sto_ext = np.sort(loss_sto_ext)
        
        cdf = np.arange(1, len(sorted_trad) + 1) / len(sorted_trad)
        
        axes[0, 1].plot(sorted_trad, cdf, label='기존 PF', linewidth=2, color='blue')
        axes[0, 1].plot(sorted_sto_fin, cdf, label='STO PF (금융)', linewidth=2, color='orange')
        axes[0, 1].plot(sorted_sto_ext, cdf, label='STO PF (금융+가계)', linewidth=2, color='red', linestyle='--')
        axes[0, 1].axhline(0.95, color='gray', linestyle=':', alpha=0.7)
        axes[0, 1].axhline(0.99, color='gray', linestyle=':', alpha=0.7)
        axes[0, 1].set_xlabel('누적 손실 (억원)', fontsize=11)
        axes[0, 1].set_ylabel('누적 확률', fontsize=11)
        axes[0, 1].set_title('누적 분포 함수 (CDF)', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 개인 투자자 손실
        retail_loss = self.results_sto['losses']['retail_loss'].sum(axis=(1, 2))
        
        axes[1, 0].hist(retail_loss, bins=50, alpha=0.7, color='red', density=True)
        axes[1, 0].set_xlabel('개인 투자자 누적 손실 (억원)', fontsize=11)
        axes[1, 0].set_ylabel('확률 밀도', fontsize=11)
        axes[1, 0].set_title('STO 개인 투자자 손실 분포', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # 손실 분해
        sec_loss_sto = self.results_sto['losses']['securities_loss'].sum(axis=(1, 2))
        con_loss_sto = self.results_sto['losses']['construction_loss'].sum(axis=(1, 2))
        
        axes[1, 1].hist([sec_loss_sto, con_loss_sto, retail_loss], bins=40, 
                       label=['증권사', '시공사', '개인'], alpha=0.6, density=True)
        axes[1, 1].set_xlabel('누적 손실 (억원)', fontsize=11)
        axes[1, 1].set_ylabel('확률 밀도', fontsize=11)
        axes[1, 1].set_title('STO PF - 손실 주체별 분포', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_time_evolution(self, save_path: str = None):
        """시간별 진화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        time = np.arange(self.params.T)
        
        # 분양률 진화
        sales_trad = self.results_trad['state']['sales_rate'].mean(axis=(0, 1))
        sales_sto = self.results_sto['state']['sales_rate'].mean(axis=(0, 1))
        
        axes[0, 0].plot(time, sales_trad, label='기존 PF', linewidth=2, marker='o', markersize=4)
        axes[0, 0].plot(time, sales_sto, label='STO PF', linewidth=2, marker='s', markersize=4)
        axes[0, 0].set_xlabel('시간 (분기)', fontsize=11)
        axes[0, 0].set_ylabel('평균 분양률', fontsize=11)
        axes[0, 0].set_title('분양률 진화', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 상관계수 진화
        corr_trad = self.results_trad['state']['correlation']
        corr_sto = self.results_sto['state']['correlation']
        
        axes[0, 1].plot(time, corr_trad.mean(axis=0), label='기존 PF', linewidth=2)
        axes[0, 1].fill_between(time,
                               np.percentile(corr_trad, 25, axis=0),
                               np.percentile(corr_trad, 75, axis=0),
                               alpha=0.3)
        axes[0, 1].plot(time, corr_sto.mean(axis=0), label='STO PF', linewidth=2)
        axes[0, 1].fill_between(time,
                               np.percentile(corr_sto, 25, axis=0),
                               np.percentile(corr_sto, 75, axis=0),
                               alpha=0.3)
        axes[0, 1].set_xlabel('시간 (분기)', fontsize=11)
        axes[0, 1].set_ylabel('상관계수 ρ(t)', fontsize=11)
        axes[0, 1].set_title('프로젝트 간 상관계수 진화', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 시스템 손실 진화
        loss_trad = self.results_trad['losses']['systemic_loss']
        loss_sto = self.results_sto['losses']['systemic_loss']
        
        axes[1, 0].plot(time, loss_trad.mean(axis=0), label='기존 PF', linewidth=2)
        axes[1, 0].plot(time, loss_sto.mean(axis=0), label='STO PF', linewidth=2)
        axes[1, 0].set_xlabel('시간 (분기)', fontsize=11)
        axes[1, 0].set_ylabel('평균 손실 (억원)', fontsize=11)
        axes[1, 0].set_title('시스템 손실 진화', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 소비 충격 (STO만)
        consumption = self.results_sto['state']['consumption_shock']
        
        axes[1, 1].plot(time, consumption.mean(axis=0), linewidth=2, color='red')
        axes[1, 1].fill_between(time,
                               np.percentile(consumption, 25, axis=0),
                               np.percentile(consumption, 75, axis=0),
                               alpha=0.3, color='red')
        axes[1, 1].set_xlabel('시간 (분기)', fontsize=11)
        axes[1, 1].set_ylabel('소비 충격', fontsize=11)
        axes[1, 1].set_title('STO - 소비 위축 피드백', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_metrics_comparison(self, analyzer_trad, analyzer_sto, save_path: str = None):
        """리스크 지표 비교"""
        m_trad = analyzer_trad.metrics
        m_sto = analyzer_sto.metrics
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # VaR/ES 비교
        categories = ['VaR 95%', 'ES 95%', 'VaR 99%', 'ES 99%']
        trad_vals = [m_trad['VaR_95'], m_trad['ES_95'], m_trad['VaR_99'], m_trad['ES_99']]
        sto_vals = [m_sto['VaR_95'], m_sto['ES_95'], m_sto['VaR_99'], m_sto['ES_99']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, trad_vals, width, label='기존 PF', color='blue', alpha=0.7)
        bars2 = axes[0, 0].bar(x + width/2, sto_vals, width, label='STO PF', color='orange', alpha=0.7)
        axes[0, 0].set_ylabel('손실 (억원)', fontsize=11)
        axes[0, 0].set_title('Tail Risk 비교 (금융권)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(categories)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # 동시 부실 확률
        default_cats = ['10% 이상', '20% 이상', '30% 이상']
        trad_default = [
            m_trad['prob_10pct_simultaneous'],
            m_trad['prob_20pct_simultaneous'],
            m_trad['prob_30pct_simultaneous'],
        ]
        sto_default = [
            m_sto['prob_10pct_simultaneous'],
            m_sto['prob_20pct_simultaneous'],
            m_sto['prob_30pct_simultaneous'],
        ]
        
        x2 = np.arange(len(default_cats))
        axes[0, 1].bar(x2 - width/2, trad_default, width, label='기존 PF', color='blue', alpha=0.7)
        axes[0, 1].bar(x2 + width/2, sto_default, width, label='STO PF', color='orange', alpha=0.7)
        axes[0, 1].set_ylabel('확률', fontsize=11)
        axes[0, 1].set_title('동시 부실 확률', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x2)
        axes[0, 1].set_xticklabels(default_cats)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # 개인 투자자 vs 금융권
        loss_cats = ['증권사', '시공사', '개인']
        sec_mean = self.results_sto['losses']['securities_loss'].sum(axis=(1, 2)).mean()
        con_mean = self.results_sto['losses']['construction_loss'].sum(axis=(1, 2)).mean()
        retail_mean = self.results_sto['losses']['retail_loss'].sum(axis=(1, 2)).mean()
        
        axes[1, 0].bar(loss_cats, [sec_mean, con_mean, retail_mean], 
                      color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('평균 누적 손실 (억원)', fontsize=11)
        axes[1, 0].set_title('STO PF - 손실 주체별 평균', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # 전이 속도
        speed_cats = ['평균', '중앙값', '5th (빠름)']
        trad_speed = [
            m_trad['mean_contagion_time'],
            m_trad['median_contagion_time'],
            m_trad['contagion_speed_5th'],
        ]
        sto_speed = [
            m_sto['mean_contagion_time'],
            m_sto['median_contagion_time'],
            m_sto['contagion_speed_5th'],
        ]
        
        x3 = np.arange(len(speed_cats))
        axes[1, 1].bar(x3 - width/2, trad_speed, width, label='기존 PF', color='blue', alpha=0.7)
        axes[1, 1].bar(x3 + width/2, sto_speed, width, label='STO PF', color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('시간 (분기)', fontsize=11)
        axes[1, 1].set_title('전이 속도', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x3)
        axes[1, 1].set_xticklabels(speed_cats)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    print("✓ 개선된 분석 모듈 준비 완료")