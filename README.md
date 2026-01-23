Multi-Asset Unsupervised Market Regime Discovery
Discover latent market states in cryptocurrency markets without labels or predictions—only pure market state awareness.





⚠️ DISCLAIMER: This is a personal research project for educational and entertainment purposes only. It is NOT production-ready, NOT financial advice, and should NOT be used for real trading decisions. Markets are complex and unpredictable—this system is designed for learning, not profit.

🎯 Project Overview
This system identifies recurring structural states ("regimes") in cryptocurrency markets using purely unsupervised deep learning. Instead of making predictions, it provides asset-specific probabilistic context (P(up | regime)) with temporal duration and volatility awareness for BTC, ETH, and BNB.

Key Philosophy: Understand market states, don't predict prices.

🔬 Learning Process & Methodology
Phase 1: Single-Asset Foundation (BTC Only)
Initial Approach: Trained autoencoder on BTC data exclusively
Silhouette Score: 0.455 (excellent cluster separation)
Limitation: Could not capture cross-asset dynamics or relative strength relationships
Insight: While technically successful, single-asset approach missed crucial market structure
Phase 2: Multi-Asset Integration Challenge
Hypothesis: Combining assets would reveal universal market structures while preserving asset-specific behaviors
Technical Challenge: Different volatility scales, missing cross-asset features, inconsistent feature dimensions
Key Innovation: Global feature scaling across all assets + cross-asset feature engineering
Phase 3: Cross-Asset Feature Engineering
Relative Strength Features: ETH_vs_btc, BNB_vs_btc
Correlation Features: ETH_btc_corr, BNB_btc_corr
Volatility Spreads: ETH_vol_spread, BNB_vol_spread
Global Scaling: Single StandardScaler fitted on concatenated asset data
Phase 4: Multi-Asset Training & Results
Silhouette Score: 0.398 (slight decrease from 0.455)
Why the Decrease?:
Increased Complexity: Cross-asset features add dimensionality but also noise
Diverse Asset Behaviors: BTC (strategic), ETH (tactical), BNB (opportunistic) have different regime structures
Feature Trade-off: Gained cross-asset relationships at cost of some intra-asset purity
Why It's Worth It:
Financial Realism: Markets don't move in isolation – BTC dominance affects all assets
Enhanced Context: Relative strength provides actionable intelligence beyond absolute signals
Robust Generalization: Model learns universal market structures rather than asset-specific quirks
Phase 5: Validation & Stability Metrics
Transition Entropy: 0.024 (extremely stable, even better than single-asset)
Regime Persistence: Multi-asset approach actually improved temporal stability
Asset-Specific Statistics: Despite shared clustering, each asset maintains separate P(up) and duration statistics
📊 Evolution: Single vs Multi-Asset Comparison
Metric
Single-Asset (BTC)
Multi-Asset (BTC/ETH/BNB)
Interpretation
Silhouette Score
0.455
0.398
Slight decrease due to increased complexity
Transition Entropy
0.045
0.024
Improved stability with multi-asset context
Regime Count
8
10
Captures more complex market structures
Financial Relevance
Medium
High
Cross-asset dynamics provide real-world context
Actionable Intelligence
Limited
Comprehensive
Relative strength + absolute signals
Why the Silhouette Trade-off is Strategic
The 7% decrease in Silhouette Score (0.455 → 0.398) represents a deliberate and valuable trade-off:

Financial Reality Over Technical Purity: Real markets are interconnected – isolating assets creates artificial simplicity
Cross-Asset Alpha: Relative strength (ETH_vs_btc) often provides stronger signals than absolute price movements
Risk Management: Understanding how assets move together improves portfolio-level decision making
Regime Validation: Multi-asset consistency acts as natural validation – if BTC and ETH enter similar regimes simultaneously, confidence increases
🏗️ System Architecture
1234
Key Components
src/features.py: Stationary feature engineering with cross-asset relationships
src/model.py: Decoupled autoencoder with self-supervised objectives
scripts/train.py: Multi-asset training with global scaling
scripts/infer_latest.py: Asset-specific inference with fallback handling
scripts/walk_forward_validate.py: Stability monitoring and drift detection
📊 Results & Insights
Performance Metrics
Metric
Value
Significance
Silhouette Score
0.398
Good cluster separation despite multi-asset complexity
Transition Entropy
0.024
Extremely stable regimes (better than single-asset)
Regime Count
10
Captures complex market structures
Asset-Specific Regime Profiles
BTCUSDT (Strategic):

Regime 7: 50.73% P(up), 2031h duration (~85 days), Vol ↓18%
Regime 6: 51.54% P(up), 1816h duration (~76 days), Vol ↑15%
ETHUSDT (Tactical):

Regime 8: 51.21% P(up), 189h duration (~8 days), Vol ↑17%
Regime 2: 50.34% P(up), 284h duration (~12 days), Vol ↓13% (unstable)
BNBUSDT (Opportunistic):

Regime 0: 52.18% P(up), Vol ↑23% (strongest signal)
Regime 4: 49.99% P(up), 228h duration (~10 days), Vol ↓24% (unstable)
🚀 Usage
Training
python scripts/train.py

Daily Inference
python scripts/infer_latest.py --asset BTCUSDT
python scripts/infer_latest.py --asset ETHUSDT  
python scripts/infer_latest.py --asset BNBUSDT

Sample Output
✅ Current Market Regime for BTCUSDT detected:
   Regime-ID       : 7
   P(up next hour) : 50.73% (⌀ 2031.4h, Vol ↓18%)
   Historical Basis: 14215 examples

   🧠 Why This Approach Works (For Learning)
Educational Value
Financial Realism: Demonstrates how BTC dominance affects other assets
Technical Robustness: Shows solutions to real ML challenges (feature alignment, scaling)
Unsupervised Learning: Pure representation learning without labels
Research Insights
Multi-Asset Dynamics: Reveals how assets move together during different market phases
Temporal Context: Shows importance of duration modeling in financial regimes
Validation Techniques: Implements walk-forward validation for unsupervised systems
📁 Project Structure
market_regime_discovery/
├── data/raw/                 # OHLCV data (BTCUSDT_1h.csv, etc.)
├── src/
│   ├── features.py          # Feature engineering
│   ├── model.py             # Decoupled autoencoder
│   ├── windowing.py         # Global scaling with raw arrays
│   └── ...                  # Other core modules
├── scripts/
│   ├── train.py             # Multi-asset training
│   ├── infer_latest.py      # Daily inference
│   ├── walk_forward_validate.py  # Stability monitoring
│   └── monitor_and_retrain.py    # Automated drift detection
├── config/config.yaml       # Unified configuration
└── artifacts/               # Generated files (excluded from Git)

⚠️ Important Limitations
Why This Isn't Production-Ready
No Risk Management: No position sizing, stop-loss, or portfolio constraints
Limited Data: Only 4 years of hourly data (2020-2024) with known market regimes
No Transaction Costs: Ignores slippage, fees, and liquidity constraints
Overfitting Risk: Small dataset relative to model complexity
No Out-of-Sample Testing: Walk-forward validation is limited to historical data
Educational Purpose Only
Learning Tool: Designed to understand unsupervised learning in finance
Concept Demonstration: Shows how to build multi-asset regime systems
Code Experimentation: Safe environment to test ML architectures
Not Financial Advice: Absolutely should not be used for real trading
🎯 Future Extensions (For Learning)
On-Chain Integration: Active addresses, gas usage, staking metrics
Macro Indicators: VIX, DXY, interest rates for broader context
Solana Support: Additional asset with proper validation
Liquidity Features: Order book imbalance, volume profiles
💡 Key Takeaways
This project demonstrates that unsupervised learning can provide interesting market insights without relying on labels or making predictions. The evolution from single-asset (Silhouette 0.455) to multi-asset (Silhouette 0.398) represents a strategic trade-off: sacrificing some technical purity for vastly improved financial realism and actionable intelligence.

By focusing on market state awareness rather than price forecasting, it delivers:

Educational value in unsupervised representation learning
Technical insights into multi-asset feature engineering
Conceptual understanding of financial regime dynamics
Safe experimentation environment for ML architecture testing
Remember: This is a learning project, not a trading system. Markets are complex, unpredictable, and dangerous—always prioritize education over speculation.

No predictions. No labels. Pure market state awareness for educational purposes only.
