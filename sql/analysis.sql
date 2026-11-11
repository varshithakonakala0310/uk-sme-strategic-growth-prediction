-- UK SME Strategic Growth — SQL analysis (synthetic recreation dataset)
-- Table: synthetic_sme_data

-- 1. Average profitability by growth category
SELECT high_growth, COUNT(*) AS companies, ROUND(AVG(profit_margin),4) AS avg_profit_margin,
       ROUND(AVG(profit_per_employee_gbp),2) AS avg_profit_per_employee
FROM synthetic_sme_data GROUP BY high_growth;

-- 2. Top 10 SMEs by profit per employee
SELECT company_id, profit_per_employee_gbp, profit_margin, turnover_growth
FROM synthetic_sme_data ORDER BY profit_per_employee_gbp DESC LIMIT 10;

-- 3. Correlation proxy: growth quartiles vs profitability
SELECT NTILE(4) OVER (ORDER BY turnover_growth) AS growth_quartile,
       ROUND(AVG(profit_margin),4) AS avg_margin, ROUND(AVG(revenue_per_employee_gbp),2) AS avg_rev_per_emp
FROM synthetic_sme_data GROUP BY 1;

-- 4. Risk view: high volatility, low liquidity firms
SELECT company_id, turnover_volatility, liquidity_ratio, debt_to_equity
FROM synthetic_sme_data WHERE turnover_volatility > 0.3 AND liquidity_ratio < 1.0
ORDER BY debt_to_equity DESC;
