# 📐 부동산 PF 몬테카를로 시뮬레이션 - 전체 수식 (V2)

**업데이트:** Q1-Q5, Part 4.1-4.2 반영

---

# 📋 **Complete Mathematical Framework**

---

## **Notation (표기법)**

| 기호 | 의미                                              |
| ---- | ------------------------------------------------- |
| $i$  | 프로젝트 인덱스 ($i = 1, \ldots, N$)              |
| $t$  | 시간 인덱스 (분기, $t = 0, \ldots, T$)            |
| $m$  | 몬테카를로 시뮬레이션 인덱스 ($m = 1, \ldots, M$) |
| $N$  | 총 프로젝트 수 (100)                              |
| $T$  | 총 시뮬레이션 기간 (16분기)                       |
| $M$  | 몬테카를로 시행 횟수 (10,000)                     |

---

# 🏗️ **Part 1: 초기 설정 (Q1 개선)**

---

## **1.1 프로젝트 기본 파라미터 (Q1)**

$$V = 1000 \text{억원}$$

**자금 조달 구조:**
$$\epsilon = 0.05 \quad \text{(자기자본)}$$
$$\lambda = 0.70 \quad \text{(유동화)}$$
$$\gamma = 0.25 \quad \text{(시공사)}$$

**제약조건:**
$$\epsilon + \lambda + \gamma = 1.0$$

**배분:**

- 자기자본: $E = V \cdot \epsilon = 50$ 억원
- 유동화: $A = V \cdot \lambda = 700$ 억원
- 시공사: $C = V \cdot \gamma = 250$ 억원

---

## **1.2 유동화 구조**

### **기존 PF:**

$$A_i^{(0)} = 70 \text{억원}$$
$$A_i^{\text{senior},(0)} = 70 \text{억원}$$
$$A_i^{\text{junior},(0)} = 0$$

### **STO PF:**

$$A_i^{(0)} = 70 \text{억원}$$
$$A_i^{\text{senior},(0)} = 70 \times (1-\theta) = 50.4 \text{억원}$$
$$A_i^{\text{junior},(0)} = 70 \times \theta = 19.6 \text{억원}$$

- $\theta = 0.28$: 후순위 비율

---

## **1.3 시공사 책임준공 (Q2)**

$$\gamma_g = 0.20 \quad \text{(책임준공 비율)}$$

**의미:**

- 총 시공비: 25억 (전체 프로젝트의 25%)
- 책임준공 보증: 20억 (프로젝트의 20%)

---

## **1.4 시공비 비율 (Part 4.1 추가)**

$$CR_i \sim \text{Uniform}(0.50, 0.70)$$

**의미:**

- 프로젝트별 고정
- 시공비가 높을수록 회수율 하락

---

## **1.5 기타 초기값**

- 시공사 등급: $G_i \in \{0,1,2,3,4\}$
- 프로젝트 단계: $B_i^{(0)} \sim \text{Bernoulli}(0.25)$
- 초기 분양률: $S_i^{(0)} = 0.15$
- 초기 상관계수: $\rho^{(0)} = 0.18$
- 초기 급매 할인: $\delta^{(0)} = 0.5$ (Part 4.2)

---

# 📈 **Part 2: 분양률 동학 (Q5 로지스틱 곡선)**

---

## **2.1 로지스틱 분양률 (Q5)**

$$S_{i,t}^{(m)} = S_{\min} + \frac{S_{\max} - S_{\min}}{1 + \exp\left(-k_{\text{adj}}(t - t_0)\right)} + \xi_{i,t}^{(m)}$$

**파라미터:**

- $S_{\min} = 0.15$: 초기 분양률
- $S_{\max} = 0.85$: 최대 분양률
- $k = 0.5$: 기본 성장 속도
- $t_0 = 8.0$: 변곡점

---

## **2.2 소비 위축 반영**

$$k_{\text{adj}} = k \times \left(1 - 2 \times \psi_t^{(m)}\right)$$

$$\psi_t^{(m)} = \min\left(0.3 \times r_t^{\text{retail},(m)}, 0.15\right)$$

---

## **2.3 노이즈**

$$\xi_{i,t}^{(m)} = \sigma_S \left( \sqrt{\rho_t^{(m)}} Z_t^{(m)} + \sqrt{1 - \rho_t^{(m)}} \epsilon_{i,t}^{(m)} \right)$$

- $\sigma_S = 0.15$

---

## **2.4 차환 실패시 분양 중단**

$$
S_{i,t}^{(m)} = \begin{cases}
\text{Logistic}(t) + \xi & \text{if } R_{i,t-1}^{(m)} = 1 \\
S_{i,t-1}^{(m)} & \text{if } R_{i,t-1}^{(m)} = 0
\end{cases}
$$

---

# 🎲 **Part 3: 차환 확률 모델**

---

## **3.1 로지스틱 회귀**

$$P\left(R_{i,t}^{(m)} = 1\right) = \frac{1}{1 + \exp(-X_{i,t}^{(m)})}$$

$$X_{i,t}^{(m)} = \alpha_0 + \alpha_s S_{i,t}^{(m)} + \alpha_{s^2} \left(S_{i,t}^{(m)}\right)^2 + \alpha_z Z_t^{(m)} + \alpha_{\Phi} \Phi_t^{(m)} + \alpha_g G_i + \alpha_b B_i^{(t)} + \alpha_e E_{i,t}^{(m)}$$

**파라미터:**

- $\alpha_0 = 2.0$
- $\alpha_s = 6.0$
- $\alpha_{s^2} = -1.5$
- $\alpha_z = -5.0$
- $\alpha_{\Phi} = 2.5$
- $\alpha_g = -1.0$
- $\alpha_b = -1.8$
- $\alpha_e = -0.6$

---

# 💵 **Part 4: 회수율 계산 (Part 4.1, 4.2 개선)**

---

## **4.1 동적 회수율 (Part 4.1)**

$$\eta_{i,t}^{\text{effective},(m)} = \left[ \eta_{\text{base}} + \beta_s S_{i,t}^{(m)} + \beta_c c - \beta_{\text{cost}} CR_i \right] \times \delta_t^{(m)}$$

**기본 회수율:**
$$\eta_{i,t}^{\text{base},(m)} = \eta_{\text{base}} + \beta_s S_{i,t}^{(m)} + \beta_c c - \beta_{\text{cost}} CR_i + \beta_g (4 - G_i)$$

**파라미터:**

- $\eta_{\text{base}} = 0.25$: 기본 회수율
- $\beta_s = 0.4$: 분양률 계수
- $\beta_c = 0.3$: 담보 계수
- $\beta_{\text{cost}} = 0.15$: 시공비 비율 계수 (신규!)
- $\beta_g = 0.05$: 등급 계수
- $c = 0.30$: 담보 비율
- $CR_i \in [0.5, 0.7]$: 시공비 비율 (신규!)

**범위:**
$$\eta_{i,t}^{\text{base},(m)} \in [0.15, 0.80]$$

---

## **4.2 시장 연동 급매 할인 (Part 4.1, 4.2)**

$$
\delta_t^{(m)} = \begin{cases}
0.3 & \text{if Panic Mode} \\
0.5 & \text{otherwise}
\end{cases}
$$

### **Panic Mode 조건 (Part 4.2):**

$$
\text{Panic}\_t^{(m)} = \begin{cases}
1 & \text{if } \frac{\sum_{i=1}^{N} \mathbb{1}(R_{i,t}^{(m)}=0)}{N} > \tau \\
0 & \text{otherwise}
\end{cases}
$$

- $\tau = 0.15$: 공황 임계치 (15% 동시 부실)

### **Absorbing State:**

$$\text{Panic}_t^{(m)} = \text{Panic}_{t-1}^{(m)} \lor \left(\text{failure\_rate}_t > \tau\right)$$

**의미:**

- 한 번 Panic Mode 진입하면 영구 고정
- $\delta = 0.3$으로 고정 (복귀 불가)

---

## **4.3 최종 회수율**

$$\eta_{i,t}^{\text{effective},(m)} = \eta_{i,t}^{\text{base},(m)} \times \delta_t^{(m)}$$

**범위:**

- Normal: $[0.15 \times 0.5, 0.80 \times 0.5] = [7.5\%, 40\%]$
- Panic: $[0.15 \times 0.3, 0.80 \times 0.3] = [4.5\%, 24\%]$

---

# 💸 **Part 5: 손실 계산 - 기존 PF (Q1, Q2, Q4 개선)**

---

## **5.1 총 손실**

$$L_{i,t}^{\text{total},(m)} = A_{i,t-1}^{(m)} \times \left(1 - \eta_{i,t}^{\text{effective},(m)}\right) \times \left(1 - R_{i,t}^{(m)}\right)$$

---

## **5.2 Waterfall (Q4)**

### **1순위: 자기자본**

$$L_{i,t}^{\text{equity},(m)} = \min\left(L_{i,t}^{\text{total},(m)}, E\right)$$

- $E = 5$ 억원

### **2순위: 증권사**

$$L_{i,t}^{\text{sec},(m)} = \max\left(L_{i,t}^{\text{total},(m)} - E, 0\right)$$

---

## **5.3 시공사 손실 (Q2)**

$$L_{i,t}^{\text{con},(m)} = \left( L_{i,t}^{\text{completion},(m)} + L_{i,t}^{\text{guarantee},(m)} \right) \times \left(1 - R_{i,t}^{(m)}\right)$$

$$L_{i,t}^{\text{completion},(m)} = V \times \gamma_g \times \left(1 - S_{i,t}^{(m)}\right) = 100 \times 0.20 \times (1-S)$$

$$L_{i,t}^{\text{guarantee},(m)} = L_{i,t}^{\text{sec},(m)}$$

---

# 🌊 **Part 6: 손실 계산 - STO PF (Q1, Q2, Q4, Part 4.1 개선)**

---

## **6.1 총 자산 (Part 4.1 급매 할인)**

$$V_{i,t}^{\text{total},(m)} = V_{i,t}^{\text{presale},(m)} + V_{i,t}^{\text{land},(m)}$$

$$V_{i,t}^{\text{presale},(m)} = S_{i,t}^{(m)} \times 100$$

$$V_{i,t}^{\text{land},(m)} = \left(1 - S_{i,t}^{(m)}\right) \times 100 \times c \times \delta_t^{(m)}$$

**급매 할인 동적:**

- Normal: $\delta = 0.5$
- Panic: $\delta = 0.3$ (Part 4.2)

---

## **6.2 Waterfall (Q4)**

$$L_{i,t}^{\text{total},(m)} = \max\left(A_{i,t}^{\text{total},(m)} - V_{i,t}^{\text{total},(m)}, 0\right)$$

### **1순위: 자기자본**

$$L_{i,t}^{\text{equity},(m)} = \min\left(L_{i,t}^{\text{total},(m)}, E\right)$$

### **2순위: 후순위**

$$L_{i,t}^{\text{remain},(m)} = \max\left(L_{i,t}^{\text{total},(m)} - E, 0\right)$$

$$L_{i,t}^{\text{junior},(m)} = \min\left(L_{i,t}^{\text{remain},(m)}, A_{i,t-1}^{\text{junior},(m)}\right)$$

### **3순위: 선순위**

$$L_{i,t}^{\text{senior},(m)} = \max\left(L_{i,t}^{\text{remain},(m)} - A_{i,t-1}^{\text{junior},(m)}, 0\right)$$

---

## **6.3 시공사 손실 (Q2)**

$$L_{i,t}^{\text{con},(m)} = \left( L_{i,t}^{\text{completion},(m)} + L_{i,t}^{\text{senior},(m)} \right) \times \left(1 - R_{i,t}^{(m)}\right)$$

---

# 🔗 **Part 7: 동적 상관계수**

---

$$\rho_t^{(m)} = \rho_{\text{base}} + \beta_{\text{sys}} \cdot \mathbb{1}\left(r_t^{\text{sys},(m)} > \tau_{\text{sys}}\right) + \beta_{\text{sto}} \cdot r_t^{\text{retail},(m)} + \beta_{\text{liq}} \cdot \Phi_t^{(m)} + \beta_{\text{cons}} \cdot \psi_t^{(m)}$$

**파라미터:**

- $\rho_{\text{base}} = 0.18$
- $\beta_{\text{sys}} = 0.30$
- $\beta_{\text{sto}} = 0.25$
- $\beta_{\text{liq}} = -0.18$
- $\beta_{\text{cons}} = 0.18$
- $\tau_{\text{sys}} = 0.12$

**범위:**
$$\rho_t^{(m)} \in [0, 0.95]$$

---

# 📊 **Part 8: 리스크 지표**

---

## **8.1 VaR (Value at Risk)**

$$\text{VaR}_{95\%} = Q_{0.95}\left(\{L^{\text{total},(1)}, \ldots, L^{\text{total},(M)}\}\right)$$

---

## **8.2 ES (Expected Shortfall)**

$$\text{ES}_{95\%} = \mathbb{E}\left[ L^{\text{total},(m)} \mid L^{\text{total},(m)} > \text{VaR}_{95\%} \right]$$

---

## **8.3 전이 속도 (Part 4.2 반영)**

$$T_{\text{panic}}^{(m)} = \min\left\{t : \text{Panic}_t^{(m)} = 1\right\}$$

**평균 전이 시간:**
$$\mathbb{E}[T_{\text{panic}}] = \frac{1}{\sum_{m=1}^{M} \mathbb{1}(\text{Panic}^{(m)}=1)} \sum_{m: \text{Panic}^{(m)}=1} T_{\text{panic}}^{(m)}$$

---

# 📝 **파라미터 총정리**

---

| 파라미터                 | 값   | 의미                   |
| ------------------------ | ---- | ---------------------- |
| **구조 (Q1)**            |      |                        |
| $\epsilon$               | 0.05 | 자기자본 비율          |
| $\lambda$                | 0.70 | 유동화 비율            |
| $\gamma$                 | 0.25 | 시공사 비율            |
| $\gamma_g$               | 0.20 | 책임준공 비율 (Q2)     |
| **분양률 (Q5)**          |      |                        |
| $S_{\min}$               | 0.15 | 초기 분양률            |
| $S_{\max}$               | 0.85 | 최대 분양률            |
| $k$                      | 0.5  | 성장 속도              |
| $t_0$                    | 8.0  | 변곡점                 |
| **회수율 (Part 4.1)**    |      |                        |
| $\eta_{\text{base}}$     | 0.25 | 기본 회수율            |
| $\beta_s$                | 0.4  | 분양률 계수            |
| $\beta_c$                | 0.3  | 담보 계수              |
| $\beta_{\text{cost}}$    | 0.15 | 시공비 계수 (신규!)    |
| **급매 할인 (Part 4.2)** |      |                        |
| $\delta_{\text{base}}$   | 0.5  | 기본 급매 할인         |
| $\delta_{\text{panic}}$  | 0.3  | 공황 급매 할인 (신규!) |
| $\tau_{\text{panic}}$    | 0.15 | 공황 임계치 (신규!)    |

---

**END - V2 완료**
