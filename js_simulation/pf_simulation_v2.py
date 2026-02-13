"""
부동산 PF 몬테카를로 시뮬레이션 (개선 버전)
- 기존 PF vs STO 도입 PF 비교
- 현실적인 차환 함수, 손실 함수, 피드백 루프 구현
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class ProjectStage(Enum):
    """프로젝트 단계"""
    BRIDGE = 0      # 브릿지론
    MAIN_PF = 1     # 본PF
    PRESALE = 2     # 분양중
    COMPLETE = 3    # 준공

class ConstructionGrade(Enum):
    """시공사 신용등급"""
    AAA = 0  # 초우량
    AA = 1   # 우량
    A = 2    # 양호
    BBB = 3  # 보통
    BB = 4   # 주의


@dataclass
class SimulationParams:
    """시뮬레이션 파라미터"""
    # 기본 설정
    n_projects: int = 100
    T: int = 16  # 4년 (분기별)
    n_simulations: int = 10000
    debug_mode: bool = False  # 디버그 모드 추가
    
    # 분양률 파라미터 (Q5: 로지스틱 곡선)
    use_logistic_sales: bool = True   # 로지스틱 곡선 사용 여부
    sales_max: float = 0.85           # 최대 분양률
    sales_growth_rate: float = 0.5    # 성장 속도 (k)
    sales_inflection: float = 8.0     # 변곡점 시점 (t_0)
    
    # 기존 선형 증가 파라미터 (use_logistic_sales=False일 때)
    mu_sales_base: float = 0.04  # 기본 분양률 증가 (0.06 → 0.04 하향)
    sigma_sales: float = 0.15    # 분양률 변동성 (0.12 → 0.15 상향)
    
    # 상관계수 파라미터
    rho_base: float = 0.18           # 기본 상관계수 (0.25 → 0.18 하향)
    beta_systemic: float = 0.30      # 시스템 손실 증가시 상관계수 증가 (0.20 → 0.30 강화!)
    beta_sto: float = 0.25           # 개인 손실 발생시 상관계수 증가 (0.12 → 0.25 상향!)
    beta_liquidity: float = -0.18    # 유동성 증가시 상관계수 감소
    beta_consumption: float = 0.18   # 소비 위축시 상관계수 증가 (0.12 → 0.18 강화!)
    
    # 차환 성공 확률 파라미터 (로지스틱 회귀) - 개선
    alpha_0: float = 2.0              # 절편 (4.0 → 2.0 하향)
    alpha_sales: float = 6.0          # 분양률 계수 (8.0 → 6.0 하향)
    alpha_sales_sq: float = -1.5      # 분양률 제곱 (-2.0 → -1.5)
    alpha_shock: float = -5.0         # 공통 충격 계수 (-4.5 → -5.0 강화!)
    alpha_sec_capacity: float = 2.5   # 증권사 여력 계수 (3.0 → 2.5)
    alpha_construction_grade: float = -1.0  # 시공사 등급 계수 (-0.8 → -1.0)
    alpha_bridge: float = -1.8        # 브릿지론 더미 (-1.2 → -1.8)
    alpha_extension: float = -0.6     # 만기연장 횟수당 불리 (-0.3 → -0.6)
    
    # 손실 파라미터 (Part 4.1 개선)
    recovery_rate_base: float = 0.25   # 기본 회수율
    beta_sales_recovery: float = 0.4   # 분양률 계수
    beta_collateral: float = 0.3       # 담보 계수
    beta_cost: float = 0.15            # 시공비 비율 계수 (Part 4.1 추가!)
    collateral_ratio: float = 0.30     # 담보 가치 비율
    
    # 시공비 비율 범위 (Part 4.1)
    construction_cost_min: float = 0.50  # 최소 시공비 비율
    construction_cost_max: float = 0.70  # 최대 시공비 비율
    
    # 급매 할인 (Part 4.1, 4.2)
    fire_sale_base: float = 0.5        # 기본 급매 할인 (50%)
    fire_sale_panic: float = 0.3       # 공황 급매 할인 (30%, absorbing state)
    panic_threshold: float = 0.15      # 공황 임계치 (15% 동시 부실)
    
    # 시공사 책임준공 비용
    construction_guarantee_ratio: float = 0.20  # 시공사 책임준공 비율 (20%)
    
    # 프로젝트 단계별 리스크 가중치
    stage_risk_weight: Dict[ProjectStage, float] = None
    
    # 시공사 등급별 분포
    construction_grade_dist: List[float] = None
    
    # ===== 자금 조달 구조 (Funding Structure) =====
    # 국내 PF 실무: 자기자본 3-5%, 부채 95-97%
    equity_ratio: float = 0.05         # 자기자본 비율 (5%)
    debt_ratio: float = 0.95           # 부채(유동화) 비율 (95%)
    sto_ratio: float = 0.28            # 후순위(Junior/STO) 비율 (부채 중)
    
    # ===== 비용 구조 (Cost Structure, 참고용) =====
    # 주의: 자금 조달과 별개 (지출 항목)
    construction_cost_ratio: float = 0.60   # 시공비 60%
    land_cost_ratio: float = 0.25           # 토지비 25%
    other_cost_ratio: float = 0.15          # 기타비 15%
    
    # 시스템 리스크 임계치
    systemic_threshold: float = 0.12
    
    # 초기값
    initial_sales: float = 0.15
    total_project_value: float = 1000.0  # 억원 (현실성 반영)
    
    def __post_init__(self):
        """기본값 설정 및 검증"""
        if self.stage_risk_weight is None:
            self.stage_risk_weight = {
                ProjectStage.BRIDGE: 1.0,
                ProjectStage.MAIN_PF: 0.65,
                ProjectStage.PRESALE: 0.35,
                ProjectStage.COMPLETE: 0.08,
            }
        
        if self.construction_grade_dist is None:
            # AAA: 10%, AA: 25%, A: 35%, BBB: 20%, BB: 10%
            self.construction_grade_dist = [0.10, 0.25, 0.35, 0.20, 0.10]
        
        # 검증: 자기자본 + 부채 = 100%
        total_funding = self.equity_ratio + self.debt_ratio
        assert abs(total_funding - 1.0) < 1e-6, \
            f"자금 조달 비율 합계가 100%가 아닙니다: {total_funding:.2%} (자기자본 {self.equity_ratio:.1%} + 부채 {self.debt_ratio:.1%})"
        
        # 참고: 비용 구조는 별도 (합계 검증 불필요)
        # construction_cost + land_cost + other_cost = 100% (지출)


class ImprovedPFSimulation:
    """개선된 부동산 PF 시뮬레이션"""
    
    def __init__(self, params: SimulationParams, use_sto: bool = False):
        self.params = params
        self.use_sto = use_sto
        self.results = None
        
    def initialize_state(self) -> Dict[str, np.ndarray]:
        """상태 변수 초기화"""
        p = self.params
        
        state = {
            # 기본 상태
            'sales_rate': np.zeros((p.n_simulations, p.n_projects, p.T)),
            'unsold_amount': np.zeros((p.n_simulations, p.n_projects, p.T)),
            'project_stage': np.zeros((p.n_simulations, p.n_projects, p.T), dtype=int),
            
            # 재무 상태
            'securitization_balance': np.zeros((p.n_simulations, p.n_projects, p.T)),
            'securitization_senior': np.zeros((p.n_simulations, p.n_projects, p.T)),
            'securitization_junior': np.zeros((p.n_simulations, p.n_projects, p.T)),
            
            # 차환 및 연장
            'refinance_success': np.zeros((p.n_simulations, p.n_projects, p.T), dtype=bool),
            'extension_count': np.zeros((p.n_simulations, p.n_projects, p.T), dtype=int),
            
            # 시장 변수
            'common_shock': np.zeros((p.n_simulations, p.T)),
            'securities_capacity': np.ones((p.n_simulations, p.T)),
            'correlation': np.zeros((p.n_simulations, p.T)),
            
            # 거시경제 피드백
            'consumption_shock': np.zeros((p.n_simulations, p.T)),  # 소비 위축
            'credit_tightening': np.zeros((p.n_simulations, p.T)),  # 신용 경색
        }
        
        # 시공사 등급 할당 (프로젝트별 고정)
        construction_grades = np.random.choice(
            list(ConstructionGrade),
            size=(p.n_simulations, p.n_projects),
            p=p.construction_grade_dist
        )
        state['construction_grade'] = np.array([[g.value for g in row] for row in construction_grades])
        
        # Part 4.1: 시공비 비율 (프로젝트별 고정, 0.5~0.7)
        state['construction_cost_ratio'] = np.random.uniform(
            p.construction_cost_min,
            p.construction_cost_max,
            size=(p.n_simulations, p.n_projects)
        )
        
        # Part 4.2: 급매 할인율 상태 (absorbing state 추적)
        state['fire_sale_discount'] = np.ones((p.n_simulations, p.T)) * p.fire_sale_base
        state['panic_mode'] = np.zeros((p.n_simulations, p.T), dtype=bool)
        
        # 초기 분양률
        state['sales_rate'][:, :, 0] = p.initial_sales
        
        # 초기 프로젝트 단계: 모두 본PF로 시작 (단순화)
        state['project_stage'][:, :, 0] = ProjectStage.MAIN_PF.value
        
        # 초기 유동화 잔액 (부채 95%)
        initial_debt = p.total_project_value * p.debt_ratio
        state['securitization_balance'][:, :, 0] = initial_debt
        
        if self.use_sto:
            # STO 구조: Senior + Junior
            state['securitization_senior'][:, :, 0] = initial_debt * (1 - p.sto_ratio)
            state['securitization_junior'][:, :, 0] = initial_debt * p.sto_ratio
        else:
            # 기존 PF: Senior만
            state['securitization_senior'][:, :, 0] = initial_debt
        
        # 초기 미분양
        state['unsold_amount'][:, :, 0] = (1 - p.initial_sales) * p.total_project_value
        
        # 초기 상관계수
        state['correlation'][:, 0] = p.rho_base
        
        return state
    
    def update_correlation(self, t: int, state: Dict[str, np.ndarray],
                          systemic_loss_prev: np.ndarray,
                          losses: Dict[str, np.ndarray]) -> np.ndarray:
        """
        동적 상관계수 업데이트
        
        ρ(t) = ρ_base + β_sys·I(L_sys > θ) + β_sto·Retail_Loss_Ratio
               + β_liq·Capacity + β_cons·Consumption_shock
        """
        p = self.params
        
        rho = np.ones(p.n_simulations) * p.rho_base
        
        # 시스템 손실 효과
        total_value = p.n_projects * p.total_project_value
        loss_ratio = systemic_loss_prev / total_value
        rho += p.beta_systemic * (loss_ratio > p.systemic_threshold)
        
        # STO 효과 - 개인 손실 발생시에만 상관계수 증가 (개선!)
        if self.use_sto and t > 1 and 'retail_loss' in losses:
            # 이전 시점 개인 손실률 계산
            retail_loss_prev = losses['retail_loss'][:, :, t-1].sum(axis=1)
            retail_exposure = state['securitization_junior'][:, :, 0].sum(axis=1)
            retail_loss_ratio = retail_loss_prev / (retail_exposure + 1e-10)
            
            # 손실이 발생했을 때만 상관계수 증가 (군집 행동)
            rho += p.beta_sto * retail_loss_ratio
        
        # 유동성 효과
        rho += p.beta_liquidity * state['securities_capacity'][:, t-1]
        
        # 소비 위축 효과
        if self.use_sto:
            rho += p.beta_consumption * state['consumption_shock'][:, t-1]
        
        return np.clip(rho, 0, 0.95)
    
    def simulate_sales_rate(self, t: int, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        분양률 시뮬레이션 (Q5: 로지스틱 곡선)
        
        - 로지스틱 함수: 초기 폭발 → 중기 증가 → 후기 정체
        - 소비 위축에 따라 성장 둔화
        - 차환 실패시 분양 중단
        """
        p = self.params
        
        if p.use_logistic_sales:
            # Q5: 로지스틱 곡선 (S자)
            # S(t) = S_min + (S_max - S_min) / (1 + exp(-k(t - t_0)))
            
            # 기본 트렌드
            S_min = p.initial_sales  # 0.15
            S_max = p.sales_max      # 0.85
            k = p.sales_growth_rate  # 0.5
            t_0 = p.sales_inflection # 8.0
            
            # 소비 위축에 따른 성장 속도 조정
            consumption_shock_avg = state['consumption_shock'][:, t].mean()
            k_adjusted = k * (1 - consumption_shock_avg * 2)  # 소비 위축시 성장 둔화
            
            # 로지스틱 곡선 계산
            logistic_trend = S_min + (S_max - S_min) / (
                1 + np.exp(-k_adjusted * (t - t_0))
            )
            
            # 노이즈 추가 (개별 + 공통)
            Z = state['common_shock'][:, t]
            epsilon = np.random.randn(p.n_simulations, p.n_projects)
            rho = state['correlation'][:, t]
            
            noise = p.sigma_sales * (
                np.sqrt(rho)[:, np.newaxis] * Z[:, np.newaxis] +
                np.sqrt(1 - rho)[:, np.newaxis] * epsilon
            )
            
            # 트렌드 + 노이즈 (브로드캐스팅)
            # logistic_trend는 스칼라, noise는 (n_sim, n_proj)
            sales_rate = logistic_trend + noise
            
        else:
            # 기존: 선형 증가
            mu_adjusted = p.mu_sales_base - state['consumption_shock'][:, t]
            
            Z = state['common_shock'][:, t]
            epsilon = np.random.randn(p.n_simulations, p.n_projects)
            rho = state['correlation'][:, t]
            
            delta_sales = (
                mu_adjusted[:, np.newaxis] +
                p.sigma_sales * (
                    np.sqrt(rho)[:, np.newaxis] * Z[:, np.newaxis] +
                    np.sqrt(1 - rho)[:, np.newaxis] * epsilon
                )
            )
            
            sales_rate = state['sales_rate'][:, :, t-1] + delta_sales
        
        # 차환 실패시 분양 중단
        if t > 1:
            prev_fail = ~state['refinance_success'][:, :, t-1]
            # 차환 실패시 이전 시점 분양률로 고정
            failed_sales = state['sales_rate'][:, :, t-1]
            sales_rate = np.where(prev_fail, failed_sales, sales_rate)
        
        return np.clip(sales_rate, 0, 1)
    
    def calculate_recovery_rate(self, t: int, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        회수율 계산 (Part 4.1 개선)
        
        η_effective = [η_base + β_s*S + β_c*c - β_cost*CR] × δ(Z)
        
        Returns:
            base_recovery: 기본 회수율
            effective_recovery: 급매 할인 적용 회수율
        """
        p = self.params
        
        # 기본 회수율
        recovery = np.ones((p.n_simulations, p.n_projects)) * p.recovery_rate_base
        
        # 분양률 효과 (Part 4.1)
        # state['sales_rate']는 이미 2차원 배열 (n_simulations, n_projects)
        sales_effect = state['sales_rate'] * p.beta_sales_recovery
        
        # 담보 가치 효과 (Part 4.1)
        collateral_effect = p.collateral_ratio * p.beta_collateral
        
        # 시공비 비율 효과 (Part 4.1 추가!)
        cost_penalty = state['construction_cost_ratio'] * p.beta_cost
        
        # 시공사 등급 효과
        grade_effect = (4 - state['construction_grade']) * 0.05
        
        # 기본 회수율 계산
        base_recovery = recovery + sales_effect + collateral_effect + grade_effect - cost_penalty
        base_recovery = np.clip(base_recovery, 0.15, 0.80)
        
        # Part 4.1: 시장 연동 급매 할인 δ(Z_t)
        # state['fire_sale_discount']는 1차원 배열 (n_simulations,)
        fire_sale_discount = state['fire_sale_discount']
        
        # Effective 회수율 = 기본 회수율 × 급매 할인
        effective_recovery = base_recovery * fire_sale_discount[:, np.newaxis]
        
        return base_recovery, effective_recovery
        
        return base_recovery, effective_recovery
    
    def calculate_refinance_probability(self, t: int, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        차환 성공 확률 계산 (개선 버전)
        
        비선형 효과, 시공사 등급, 프로젝트 단계, 만기연장 횟수 반영
        """
        p = self.params
        
        S = state['sales_rate'][:, :, t]
        Z = state['common_shock'][:, t][:, np.newaxis]
        L_sec = state['securities_capacity'][:, t][:, np.newaxis]
        grade = state['construction_grade']  # [n_sim, n_proj]
        stage = state['project_stage'][:, :, t]
        extension = state['extension_count'][:, :, t]
        
        # 브릿지론 더미
        is_bridge = (stage == ProjectStage.BRIDGE.value).astype(float)
        
        # 로지스틱 회귀
        X = (
            p.alpha_0 +
            p.alpha_sales * S +
            p.alpha_sales_sq * (S ** 2) +  # 비선형
            p.alpha_shock * Z +
            p.alpha_sec_capacity * L_sec +
            p.alpha_construction_grade * grade +
            p.alpha_bridge * is_bridge +
            p.alpha_extension * extension
        )
        
        prob = 1 / (1 + np.exp(-X))
        return prob
    
    def simulate_refinance(self, t: int, state: Dict[str, np.ndarray]) -> np.ndarray:
        """차환 성공 여부 시뮬레이션"""
        prob = self.calculate_refinance_probability(t, state)
        uniform = np.random.rand(self.params.n_simulations, self.params.n_projects)
        return uniform < prob
    
    def update_fire_sale_discount(self, t: int, state: Dict[str, np.ndarray], p: SimulationParams):
        """
        Part 4.2: 전이 속도 및 동시 부실 반영 (Jump Process)
        
        동시 부실 비율이 임계치(τ)를 넘으면:
        - 시장 공포 → 급매 할인 강제로 0.3 이하 고정 (Absorbing State)
        - 한 번 진입하면 복귀 불가
        """
        # 동시 부실 비율 계산
        n_failures = (~state['refinance_success'][:, :, t]).sum(axis=1)  # 시뮬레이션별
        failure_rate = n_failures / p.n_projects
        
        # 임계치 초과 여부 (τ = panic_threshold, 기본 15%)
        panic_trigger = failure_rate > p.panic_threshold
        
        # Absorbing State: 한 번 panic mode 진입하면 복귀 불가
        if t > 1:
            # 이전에 이미 panic이었다면 계속 유지
            panic_trigger = panic_trigger | state['panic_mode'][:, t-1]
        
        state['panic_mode'][:, t] = panic_trigger
        
        # 급매 할인율 업데이트
        # Panic mode: 0.3 고정 (absorbing state)
        # Normal mode: 0.5 (기본)
        state['fire_sale_discount'][:, t] = np.where(
            panic_trigger,
            p.fire_sale_panic,  # 0.3
            p.fire_sale_base    # 0.5
        )
    
    def calculate_losses_traditional(self, t: int, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        기존 PF 손실 계산 (Q1, Q2, Q4, Part 4.1 반영)
        
        - Q1: 자기자본 5% 추가
        - Q2: 시공사 책임준공 20%로 조정
        - Q4: 자기자본이 First Loss 흡수
        - Part 4.1: 동적 회수율 (시공비 비율 + 시장 연동)
        """
        p = self.params
        
        refinance_fail = ~state_dict['refinance_success']
        
        # Part 4.1: 동적 회수율 계산
        base_recovery, effective_recovery = self.calculate_recovery_rate(t, state_dict)
        
        # 총 손실 계산
        total_claim = state_dict['securitization_balance']
        total_loss = total_claim * (1 - effective_recovery) * refinance_fail
        
        # Q4: 자기자본이 First Loss 흡수
        equity_amount = p.total_project_value * p.equity_ratio  # 5억
        
        # 자기자본 손실 (먼저 흡수)
        equity_loss = np.minimum(total_loss, equity_amount)
        
        # 증권사 손실 (자기자본 초과 손실)
        securities_loss = np.maximum(total_loss - equity_amount, 0)
        
        # Q2: 시공사 손실 (책임준공 20%)
        # 1) 책임준공 비용
        remaining_construction = (
            p.total_project_value * p.construction_guarantee_ratio *
            (1 - state_dict['sales_rate'])
        )
        
        # 2) 연대보증 손실 (증권사 손실분만)
        guarantee_loss = securities_loss
        
        construction_loss = (remaining_construction + guarantee_loss) * refinance_fail
        
        return {
            'equity_loss': equity_loss,        # Q4: 자기자본 손실 추가
            'securities_loss': securities_loss,
            'construction_loss': construction_loss,
        }
    
    def calculate_losses_sto(self, t: int, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        STO PF 손실 계산 (Q1, Q2, Q4 개선 반영)
        
        - Q1: 자기자본 5% 추가
        - Q2: 시공사 책임준공 20%로 조정
        - Q4: Waterfall 순서 = 자기자본 → 후순위 → 선순위
        - 급매 할인(Fire-sale discount) 반영
        """
        p = self.params
        
        refinance_fail = ~state_dict['refinance_success']
        
        # Part 4.1: 총 자산 = 분양 수입 + 담보 가치 (동적 급매 할인)
        # state_dict['fire_sale_discount']는 1차원 배열 (n_simulations,)
        fire_sale_discount = state_dict['fire_sale_discount'][:, np.newaxis]
        
        presale_revenue = state_dict['sales_rate'] * p.total_project_value
        
        # 미분양 토지의 담보 가치 (급매 할인 + 기존 담보 비율)
        land_value = (
            (1 - state_dict['sales_rate']) * 
            p.total_project_value * 
            p.collateral_ratio *
            fire_sale_discount
        )
        
        total_asset = presale_revenue + land_value
        
        # 총 부채
        total_claim = (
            state_dict['securitization_senior'] + 
            state_dict['securitization_junior']
        )
        
        # 총 손실
        total_loss = np.maximum(total_claim - total_asset, 0) * refinance_fail
        
        # Q4: Waterfall 청산 순서
        equity_amount = p.total_project_value * p.equity_ratio  # 5억
        
        # 1순위: 자기자본 손실
        equity_loss = np.minimum(total_loss, equity_amount)
        
        # 잔여 손실
        remaining_loss = np.maximum(total_loss - equity_amount, 0)
        
        # 2순위: 후순위 손실 (Junior - First Loss after Equity)
        junior_claim = state_dict['securitization_junior']
        junior_loss = np.minimum(remaining_loss, junior_claim)
        
        # 3순위: 선순위 손실 (Senior - Last Loss)
        senior_claim = state_dict['securitization_senior']
        senior_loss = np.maximum(remaining_loss - junior_claim, 0)
        
        # Q2: 시공사 손실 (책임준공 20%)
        remaining_construction = (
            p.total_project_value * p.construction_guarantee_ratio *
            (1 - state_dict['sales_rate'])
        )
        construction_loss = (remaining_construction + senior_loss) * refinance_fail
        
        return {
            'equity_loss': equity_loss,        # Q4: 자기자본 손실 추가
            'securities_loss': senior_loss,
            'construction_loss': construction_loss,
            'retail_loss': junior_loss,
        }
    
    def update_macro_feedback(self, t: int, state: Dict[str, np.ndarray],
                             losses: Dict[str, np.ndarray]):
        """
        거시경제 피드백 업데이트
        
        - 개인 손실 → 소비 위축
        - 금융기관 손실 → 신용 경색
        """
        p = self.params
        
        if self.use_sto and 'retail_loss' in losses:
            # 개인 손실 → 소비 위축
            total_retail_loss = losses['retail_loss'].sum(axis=1)
            total_retail_exposure = (
                state['securitization_junior'][:, :, 0].sum(axis=1)
            )
            
            # 손실률이 높을수록 소비 위축
            retail_loss_ratio = total_retail_loss / (total_retail_exposure + 1e-10)
            state['consumption_shock'][:, t] = np.clip(retail_loss_ratio * 0.3, 0, 0.15)
        
        # 금융기관 손실 → 신용 경색
        total_financial_loss = (
            losses['securities_loss'].sum(axis=1) +
            losses['construction_loss'].sum(axis=1)
        )
        total_value = p.n_projects * p.total_project_value
        financial_loss_ratio = total_financial_loss / total_value
        
        state['credit_tightening'][:, t] = np.clip(financial_loss_ratio, 0, 0.5)
    
    def update_project_stage(self, t: int, state: Dict[str, np.ndarray]):
        """
        프로젝트 단계 전환
        
        브릿지론 → 본PF → 분양중 → 준공
        """
        p = self.params
        
        current_stage = state['project_stage'][:, :, t-1]
        sales = state['sales_rate'][:, :, t]
        refinance = state['refinance_success'][:, :, t]
        
        new_stage = current_stage.copy()
        
        # 브릿지론 → 본PF (차환 성공 + 분양률 20% 이상)
        bridge_to_pf = (
            (current_stage == ProjectStage.BRIDGE.value) &
            refinance &
            (sales >= 0.2)
        )
        new_stage[bridge_to_pf] = ProjectStage.MAIN_PF.value
        
        # 본PF → 분양중 (분양률 50% 이상)
        pf_to_presale = (
            (current_stage == ProjectStage.MAIN_PF.value) &
            (sales >= 0.5)
        )
        new_stage[pf_to_presale] = ProjectStage.PRESALE.value
        
        # 분양중 → 준공 (분양률 85% 이상)
        presale_to_complete = (
            (current_stage == ProjectStage.PRESALE.value) &
            (sales >= 0.85)
        )
        new_stage[presale_to_complete] = ProjectStage.COMPLETE.value
        
        state['project_stage'][:, :, t] = new_stage
    
    def run_simulation(self) -> Dict[str, np.ndarray]:
        """시뮬레이션 실행"""
        p = self.params
        
        state = self.initialize_state()
        
        # 손실 변수
        if self.use_sto:
            losses = {
                'equity_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),  # Q4 추가
                'securities_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),
                'construction_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),
                'retail_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),
                'systemic_loss': np.zeros((p.n_simulations, p.T)),
                'systemic_loss_extended': np.zeros((p.n_simulations, p.T)),
            }
        else:
            losses = {
                'equity_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),  # Q4 추가
                'securities_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),
                'construction_loss': np.zeros((p.n_simulations, p.n_projects, p.T)),
                'systemic_loss': np.zeros((p.n_simulations, p.T)),
            }
        
        # 시간 루프
        for t in range(1, p.T):
            # 공통 충격
            state['common_shock'][:, t] = np.random.randn(p.n_simulations)
            
            # 상관계수 업데이트
            if t > 1:
                state['correlation'][:, t] = self.update_correlation(
                    t, state, losses['systemic_loss'][:, t-1], losses
                )
            else:
                state['correlation'][:, t] = p.rho_base
            
            # 분양률
            state['sales_rate'][:, :, t] = self.simulate_sales_rate(t, state)
            
            # 미분양
            state['unsold_amount'][:, :, t] = (
                (1 - state['sales_rate'][:, :, t]) * p.total_project_value
            )
            
            # 차환
            state['refinance_success'][:, :, t] = self.simulate_refinance(t, state)
            
            # 만기연장 횟수 (차환 실패시 +1)
            state['extension_count'][:, :, t] = (
                state['extension_count'][:, :, t-1] +
                (~state['refinance_success'][:, :, t]).astype(int)
            )
            
            # Part 4.2: 전이 속도 및 동시 부실 반영 (Jump Process)
            self.update_fire_sale_discount(t, state, p)
            
            # 손실 계산 (t-1 시점 잔액 사용) - 순서 중요!
            if self.use_sto:
                # t-1 시점 잔액으로 손실 계산
                temp_state = state.copy()
                temp_state_t = {
                    'sales_rate': state['sales_rate'][:, :, t],
                    'refinance_success': state['refinance_success'][:, :, t],
                    'securitization_senior': state['securitization_senior'][:, :, t-1],  # t-1 사용!
                    'securitization_junior': state['securitization_junior'][:, :, t-1],  # t-1 사용!
                    'construction_grade': state['construction_grade'],
                    'construction_cost_ratio': state['construction_cost_ratio'],  # Part 4.1
                    'fire_sale_discount': state['fire_sale_discount'][:, t],  # Part 4.1, 4.2
                    'correlation': state['correlation'][:, :t+1],
                }
                loss_t = self.calculate_losses_sto(t, temp_state_t)
                losses['equity_loss'][:, :, t] = loss_t['equity_loss']  # Q4 추가
                losses['securities_loss'][:, :, t] = loss_t['securities_loss']
                losses['construction_loss'][:, :, t] = loss_t['construction_loss']
                losses['retail_loss'][:, :, t] = loss_t['retail_loss']
                
                losses['systemic_loss'][:, t] = (
                    loss_t['equity_loss'].sum(axis=1) +      # 자기자본 손실
                    loss_t['securities_loss'].sum(axis=1) +
                    loss_t['construction_loss'].sum(axis=1)
                )
                losses['systemic_loss_extended'][:, t] = (
                    losses['systemic_loss'][:, t] +
                    loss_t['retail_loss'].sum(axis=1)
                    # equity_loss는 이미 포함됨 (중복 방지)
                )
            else:
                # t-1 시점 잔액으로 손실 계산
                temp_state_t = {
                    'sales_rate': state['sales_rate'][:, :, t],
                    'refinance_success': state['refinance_success'][:, :, t],
                    'securitization_balance': state['securitization_balance'][:, :, t-1],  # t-1 사용!
                    'construction_grade': state['construction_grade'],
                    'construction_cost_ratio': state['construction_cost_ratio'],  # Part 4.1
                    'fire_sale_discount': state['fire_sale_discount'][:, t],  # Part 4.1, 4.2
                    'correlation': state['correlation'][:, :t+1],
                }
                loss_t = self.calculate_losses_traditional(t, temp_state_t)
                losses['equity_loss'][:, :, t] = loss_t['equity_loss']  # Q4 추가
                losses['securities_loss'][:, :, t] = loss_t['securities_loss']
                losses['construction_loss'][:, :, t] = loss_t['construction_loss']
                
                losses['systemic_loss'][:, t] = (
                    loss_t['equity_loss'].sum(axis=1) +      # 자기자본 손실
                    loss_t['securities_loss'].sum(axis=1) +
                    loss_t['construction_loss'].sum(axis=1)
                )
            
            # 유동화 잔액 업데이트 (손실 계산 후!)
            state['securitization_balance'][:, :, t] = (
                state['securitization_balance'][:, :, t-1] *
                state['refinance_success'][:, :, t]
            )
            
            if self.use_sto:
                state['securitization_senior'][:, :, t] = (
                    state['securitization_senior'][:, :, t-1] *
                    state['refinance_success'][:, :, t]
                )
                state['securitization_junior'][:, :, t] = (
                    state['securitization_junior'][:, :, t-1] *
                    state['refinance_success'][:, :, t]
                )
            else:
                state['securitization_senior'][:, :, t] = state['securitization_balance'][:, :, t]
            
            # 거시경제 피드백
            self.update_macro_feedback(t, state, loss_t)
            
            # 증권사 여력
            total_loss = losses['securities_loss'][:, :, t].sum(axis=1)
            total_value = p.n_projects * p.total_project_value
            loss_ratio = total_loss / total_value
            state['securities_capacity'][:, t] = np.clip(
                state['securities_capacity'][:, t-1] - loss_ratio, 0, 1
            )
            
            # 프로젝트 단계 전환
            self.update_project_stage(t, state)
        
        self.results = {'state': state, 'losses': losses}
        return self.results


if __name__ == "__main__":
    print("개선된 부동산 PF 몬테카를로 시뮬레이션 - 준비 완료")
    print("\n테스트 실행...")
    
    params = SimulationParams(n_simulations=100, T=8)  # 축소 테스트
    
    print("\n기존 PF 초기화...")
    sim_trad = ImprovedPFSimulation(params, use_sto=False)
    state_trad = sim_trad.initialize_state()
    print(f"✓ 상태 변수: {len(state_trad)} 개")
    
    print("\nSTO PF 초기화...")
    sim_sto = ImprovedPFSimulation(params, use_sto=True)
    state_sto = sim_sto.initialize_state()
    print(f"✓ 상태 변수: {len(state_sto)} 개")
    
    print("\n✓ 코드 준비 완료!")