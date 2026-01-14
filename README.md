# Customer Growth and Retention

## 1. Project Overview

This project addresses a core business problem in customer growth and retention:

> If the retention budget is only sufficient to target 20% of the customer base, which 20% should be selected to maximize business value?

Instead of treating churn as a binary outcome, this project compares three different modeling: **classification, transactional models, and survival analysis** to design and evaluate retention targeting strategies under budget constraints.

The project focuses on:
- Modeling customer churn and lifetime behavior
- Estimating customer value
- Comparing strategies
- Designing decision rules for retention campaigns

## 2. Data Description

The project uses transactional customer data, including:
- `customer_id`
- `transaction_date`
- `amount`

Churn is defined as customer inactivity exceeding a fixed threshold (e.g., 90 days) and handled using right-censoring in survival analysis.

## 3. Methods Overview

Three different approaches are implemented and compared.


## 3.1 Approach 1 – Churn Classification

### Description
A supervised classification model predicts the probability that a customer will churn within a predefined prediction window (e.g., 90 days).

### Output
- `P(churn | X)`

### Retention Strategy
- Rank customers by highest churn probability
- Select top 20% as retention targets

### Strengths
- Simple and easy to implement
- Works well for operational dashboards
- Requires minimal modeling assumptions

### Limitations
- Requires a hard churn definition
- Ignores time-to-churn
- Does not account for customer value

## 3.2 Approach 2 – BG-NBD + Gamma–Gamma

### Description
This approach models customer behavior as a transaction-generating process:
- **BG-NBD** estimates:
  - Probability that a customer is still alive
  - Expected number of future transactions
- **Gamma–Gamma** estimates:
  - Expected monetary value per transaction

### Output
- `P(alive)`
- Expected future transactions
- Static CLV

### Retention Strategy
- Rank customers by lowest `P(alive)`
- Select top 20% most “at-risk” customers

### Strengths
- No hard churn labeling required
- Well-established in retail and subscription analytics
- Direct CLV estimation

### Limitations
- No information of when churn will occur
- Assumes stable purchase behavior


---

## 3.3 Approach 3 – Survival Analysis + Gamma–Gamma (Time-dependent CLV)

### Description
This approach treats churn as a time-to-event process:

- **Survival Analysis (Cox Proportional Hazards)**:
  - Models churn risk over time
  - Produces survival curves and hazard rates
- **Gamma–Gamma**:
  - Estimates expected monetary value per transaction
- These components are combined to compute **Time-dependent CLV**:
  
    > CLV(T) = ∫₀ᵀ S(t) × λ × m × d(t) dt

where:
- S(t): conditional survival probability from “now”
- λ: purchase rate
- m: expected monetary value per transaction
- d(t): discount factor


### Output
- Survival curve (conditional from now)
- Expected remaining lifetime
- Time-dependent CLV curve

### Retention Strategy
- Rank customers by remaining CLV × churn risk
- Select top 20% with highest retention priority

### Strengths
- Correctly handles censoring
- Explicitly models time-to-churn
- Best alignment with limited retention budgets

### Limitations
- Higher modeling and implementation complexity
- Requires more careful feature engineering

## 4. Final Recommendation

> **When retention budget is limited, the optimal strategy is to prioritize customers with the highest *Time-dependent CLV* and elevated churn risk, using Survival Analysis combined with Gamma–Gamma modeling.**

This approach:
- Avoids spending budget on customers with low remaining value
- Targets customers before churn occurs
- Maximizes expected return per retention

## 5. Demo

This project provides a command-line demo (CLI) that allows users to run the full customer scoring and retention pipeline.

Each command corresponds to a specific analytical capability and can be executed independently.

### 5.1 Prerequisites
- Python 3.10 is installed

- All dependencies are installed via:

```
pip install -r requirements.txt
```
### 5.2 Unified Customer Scoring
**Purpose**

Provides a holistic snapshot of a single customer, combining churn risk, survival estimates, and customer value.

**Command**
```
python demo.py score_customer <customer_id>
```
### 5.3 Churn Prediction (Classification)
**Purpose**

Predicts the probability that a customer will churn within the fixed churn window (90 days) using a supervised classification model.

**Command**
```
python demo.py predict_churn <customer_id> --horizon_days 90
```

> Note: 
> - horizon_days is provided for interface clarity.
> - The model itself is trained on a fixed 90-day churn definition.

### 5.4 Survival Analysis (Time-to-Churn)
**Purpose**

Estimates how long a customer is expected to remain active, rather than only whether they churn.

**Command**
```
python demo.py predict_survival <customer_id>
```

### 5.5 Customer Lifetime Value (CLV) Estimation

Two CLV estimation methods are available.

**CLV via BG-NBD + Gamma–Gamma:** Estimates static CLV based on expected future transactions and expected monetary value.

**Command**
```
python demo.py estimate_clv <customer_id> --method bgnbd --horizon_months 3
```

**CLV via Survival Analysis + Gamma–Gamma:** Estimates time-dependent CLV, explicitly accounting for time-to-churn and censoring.

**Command**
```
python demo.py estimate_clv <customer_id> --method survival --horizon_months 12
```

### 5.6 Retention Prioritization (In-progress Feature)
**Purpose**

Ranks customers to support budget-constrained retention decisions, such as targeting only the top 20% of customers.

There are three strategies that considerable
- Prioritizes customers with the **highest churn probability**.
- Targets customers with high expected **value and high churn risk**, using BG-NBD-based CLV.
- Targets customers with high **remaining lifetime value**, estimated via survival analysis.

**Command**
```
python demo.py rank_customers_for_retention --top_k 5 --strategy high_churn
python demo.py rank_customers_for_retention --top_k 5 --strategy high_clv_high_churn --method bgnbd --horizon_months 3 
python demo.py rank_customers_for_retention --top_k 5 --strategy high_clv_high_churn --method survival --horizon_months 12 
```

### 5.7 Example Usage

**Input**
```
python demo.py score_customer C00001
```

**Output**
```
{
  "customer_id": "C00001",
  "churn_probability": 0.3107686399056107,
  "p_alive": 0.9584913955302469,
  "expected_remaining_lifetime": 200.1382435750885,
  "clv_bgnbd": 340.4732081165075,
  "clv_survival": 903.7417460128894
}
```
---