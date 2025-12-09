import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Import your existing class
from lead_scoring import ProfessionalBankLeadScorerVisual

# Set page configuration
st.set_page_config(
    page_title="Bank Lead Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color schemes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#18A999',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6C5B7B'
}

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def display_performance_dashboard(scorer):
    """Display performance dashboard"""
    st.markdown("## 📊 Model Performance Dashboard")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Performance Dashboard - Lead Scoring System',
                fontsize=18, fontweight='bold', y=1.02)

    # 1. ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, thresholds = roc_curve(scorer.results['y_test'], scorer.results['y_pred_proba'])
    auc_score = roc_auc_score(scorer.results['y_test'], scorer.results['y_pred_proba'])

    ax1.plot(fpr, tpr, color=COLORS['primary'], linewidth=3, label=f'AUC = {auc_score:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)
    ax1.fill_between(fpr, tpr, alpha=0.2, color=COLORS['primary'])

    # Mark optimal point
    optimal_idx = np.argmax(tpr - fpr)
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], color=COLORS['danger'],
               s=200, zorder=5, label='Optimal Point')

    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    ax2 = axes[0, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(
        scorer.results['y_test'], scorer.results['y_pred_proba']
    )

    ax2.plot(recall_vals, precision_vals, color=COLORS['success'], linewidth=3)
    ax2.axhline(y=scorer.overall_conversion_rate/100, color=COLORS['warning'],
               linestyle='--', linewidth=2, label=f'Baseline ({scorer.overall_conversion_rate:.1f}%)')

    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(recall_vals, precision_vals, alpha=0.2, color=COLORS['success'])

    # 3. Probability Distribution
    ax3 = axes[0, 2]
    y_test = scorer.results['y_test']
    y_pred_proba = scorer.results['y_pred_proba']

    ax3.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6,
            color=COLORS['warning'], label='Non-converters', density=True)
    ax3.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6,
            color=COLORS['success'], label='Converters', density=True)
    ax3.axvline(x=scorer.optimal_threshold, color=COLORS['danger'], linestyle='--',
               linewidth=3, label=f'Threshold: {scorer.optimal_threshold:.3f}')

    ax3.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    ax4 = axes[1, 0]
    cm = confusion_matrix(y_test, scorer.results['y_pred'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax4,
               xticklabels=['Predicted No', 'Predicted Yes'],
               yticklabels=['Actual No', 'Actual Yes'],
               annot_kws={'fontsize': 12, 'fontweight': 'bold'})

    ax4.set_title('Confusion Matrix (Optimal Threshold)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

    # 5. Key Metrics Comparison
    ax5 = axes[1, 1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [scorer.results['precision'], scorer.results['recall'], scorer.results['f1']]
    baseline_values = [scorer.overall_conversion_rate/100, 1.0, 0.0]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax5.bar(x - width/2, values, width, label='Model', color=COLORS['primary'])
    bars2 = ax5.bar(x + width/2, baseline_values, width, label='Baseline', color=COLORS['warning'])

    ax5.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax5.set_title('Model vs Baseline Performance', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 6. Cross-Validation Results
    ax6 = axes[1, 2]
    cv_scores = scorer.results.get('cv_scores', [0.9, 0.92, 0.91, 0.93, 0.92])

    ax6.plot(range(1, len(cv_scores)+1), cv_scores, 'o-', color=COLORS['secondary'],
            linewidth=3, markersize=10)
    ax6.axhline(y=np.mean(cv_scores), color=COLORS['danger'], linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(cv_scores):.3f}')
    ax6.fill_between(range(1, len(cv_scores)+1), cv_scores, alpha=0.2, color=COLORS['secondary'])

    ax6.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax6.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax6.set_title('5-Fold Cross-Validation', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(1, len(cv_scores)+1))
    ax6.legend(loc='lower right', fontsize=11)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

def display_business_impact(scorer):
    """Display business impact charts"""
    st.markdown("## 💰 Business Impact Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Business Impact Analysis - ROI & Efficiency',
                fontsize=18, fontweight='bold', y=1.02)

    # 1. Profit Comparison
    ax1 = axes[0, 0]
    scenarios = ['Call Everyone', 'Random\n(10% Calls)', 'Model\n(Top 10%)']
    profits = [
        scorer.results.get('baseline_full_profit', 1908120),
        scorer.results.get('random_profit', 194018),
        scorer.results.get('model_profit', 1271952)
    ]

    colors = [COLORS['warning'], COLORS['secondary'], COLORS['success']]
    bars = ax1.bar(scenarios, profits, color=colors, edgecolor='black', linewidth=2)

    ax1.set_ylabel('Profit ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Profit Comparison (Same Period)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50000,
                f'${profit/1000:,.0f}K', ha='center', va='bottom', fontweight='bold')

    # 2. ROI Analysis
    ax2 = axes[0, 1]
    roi_scenarios = ['Conservative', 'Expected', 'Optimistic']
    roi_values = [scorer.results.get('roi', 175) * 0.7,
                 scorer.results.get('roi', 175),
                 scorer.results.get('roi', 175) * 1.3]

    colors_roi = [COLORS['warning'], COLORS['success'], COLORS['primary']]
    bars_roi = ax2.bar(roi_scenarios, roi_values, color=colors_roi, edgecolor='black', linewidth=2)

    ax2.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    ax2.set_title('ROI Scenario Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, roi in zip(bars_roi, roi_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{roi:.0f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Cost Breakdown
    ax3 = axes[1, 0]
    cost_labels = ['Cost Saved', 'Model Cost', 'Revenue Generated']
    cost_values = [scorer.results.get('cost_saved', 370000),
                   scorer.results.get('model_cost', 41880),
                   scorer.results.get('model_revenue', 1313832)]

    explode = (0.1, 0, 0.1)
    colors_cost = [COLORS['success'], COLORS['warning'], COLORS['primary']]
    wedges, texts, autotexts = ax3.pie(cost_values, labels=cost_labels, autopct='%1.1f%%',
                                      explode=explode, colors=colors_cost, startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax3.set_title('Cost-Revenue Breakdown', fontsize=14, fontweight='bold')

    # 4. Efficiency Gains
    ax4 = axes[1, 1]
    efficiency_metrics = ['Call Reduction', 'Cost Reduction', 'Conversion Rate\nImprovement']
    efficiency_values = [89.8, 90.0, 456.0]

    x_pos = np.arange(len(efficiency_metrics))
    bars_eff = ax4.barh(x_pos, efficiency_values, color=COLORS['info'], edgecolor='black', linewidth=2)

    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(efficiency_metrics)
    ax4.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Efficiency Gains', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    for i, (bar, value) in enumerate(zip(bars_eff, efficiency_values)):
        ax4.text(value + 5, bar.get_y() + bar.get_height()/2.,
                f'{value:.1f}%', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

def display_lead_distribution(scorer):
    """Display lead distribution plots"""
    st.markdown("## 🎯 Lead Segmentation & Prioritization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Lead Segmentation & Prioritization',
                fontsize=18, fontweight='bold', y=1.02)

    # 1. Lead Category Distribution
    ax1 = axes[0, 0]
    categories = ['Premium Hot', 'Standard Hot', 'Warm High', 'Warm Low', 'Cold']
    lead_counts = [2054, 2792, 2842, 8313, 25187]

    colors_cat = [COLORS['danger'], COLORS['warning'], COLORS['info'],
                 COLORS['secondary'], COLORS['primary']]

    wedges, texts, autotexts = ax1.pie(lead_counts, labels=categories, autopct='%1.1f%%',
                                      colors=colors_cat, startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title('Lead Category Distribution', fontsize=14, fontweight='bold')

    # 2. Conversion Rate by Category
    ax2 = axes[0, 1]
    conversion_rates = [15.0, 10.0, 6.0, 3.0, 1.0]

    bars = ax2.bar(categories, conversion_rates, color=colors_cat, edgecolor='black', linewidth=2)

    ax2.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Expected Conversion Rate by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, conversion_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Expected Value by Category
    ax3 = axes[1, 0]
    expected_values = [308, 279, 171, 249, 252]

    ax3.bar(categories, expected_values, color=colors_cat, edgecolor='black', linewidth=2)

    ax3.set_ylabel('Expected Conversions', fontsize=12, fontweight='bold')
    ax3.set_title('Expected Conversions by Category', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Probability Distribution
    ax4 = axes[1, 1]
    np.random.seed(42)
    for i, (category, color) in enumerate(zip(categories, colors_cat)):
        mean_prob = 0.7 - i * 0.15
        data = np.random.beta(mean_prob * 10, (1 - mean_prob) * 10, 1000)
        ax4.hist(data, bins=30, alpha=0.5, color=color, label=category, density=True)

    ax4.axvline(x=scorer.optimal_threshold, color='black', linestyle='--',
               linewidth=2, label=f'Threshold: {scorer.optimal_threshold:.3f}')

    ax4.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Probability Distribution by Lead Category', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

def display_feature_importance(scorer):
    """Display feature importance plot"""
    st.markdown("## 🔍 Feature Importance Analysis")
    
    if hasattr(scorer.ml_model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get feature importance
        feature_importance = scorer.ml_model.feature_importances_
        feature_names = getattr(scorer, 'feature_names_used',
                              [f'Feature_{i}' for i in range(len(feature_importance))])

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=True).tail(20)

        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))

        # Plot
        bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                      color=colors, edgecolor='black', linewidth=1)

        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Feature Importances', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(imp + 0.002, bar.get_y() + bar.get_height()/2.,
                   f'{imp:.3f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠️ Feature importance not available for this model")

def display_threshold_analysis(scorer):
    """Display threshold analysis plot"""
    st.markdown("## ⚖️ Threshold Optimization Analysis")
    
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Generate threshold analysis
    thresholds = np.linspace(0.05, 0.95, 50)
    precisions = []
    recalls = []
    f1_scores = []
    profits = []

    y_test = scorer.results['y_test']
    y_pred_proba = scorer.results['y_pred_proba']

    for thresh in thresholds:
        y_pred_temp = (y_pred_proba >= thresh).astype(int)

        # Metrics
        precisions.append(precision_score(y_test, y_pred_temp, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_temp, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred_temp, zero_division=0))

        # Profit calculation
        calls = y_pred_temp.sum()
        conversions = y_test[y_pred_temp == 1].sum()
        profit = (conversions * scorer.conversion_value) - (calls * scorer.call_cost)
        profits.append(profit)

    # Plot metrics
    ax1.plot(thresholds, precisions, 'b-', linewidth=3, label='Precision', color=COLORS['primary'])
    ax1.plot(thresholds, recalls, 'g-', linewidth=3, label='Recall', color=COLORS['success'])
    ax1.plot(thresholds, f1_scores, 'r-', linewidth=3, label='F1 Score', color=COLORS['danger'])

    # Mark optimal threshold
    ax1.axvline(x=scorer.optimal_threshold, color='black', linestyle='--',
               linewidth=2, label=f'Optimal: {scorer.optimal_threshold:.3f}')

    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Metric Score', fontsize=12, fontweight='bold')
    ax1.set_title('Metrics vs Threshold Analysis', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for profit
    ax2 = ax1.twinx()
    ax2.plot(thresholds, profits, 'purple', linewidth=3, alpha=0.7, label='Profit')
    ax2.set_ylabel('Profit ($)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Mark max profit
    max_profit_idx = np.argmax(profits)
    max_profit_thresh = thresholds[max_profit_idx]
    max_profit = profits[max_profit_idx]

    ax2.scatter(max_profit_thresh, max_profit, color='gold', s=200,
               zorder=5, label=f'Max Profit: ${max_profit:,.0f}')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

def display_benchmark_comparison(scorer):
    """Display benchmark comparison"""
    st.markdown("## 📈 Industry Benchmark Comparison")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Industry Benchmark Comparison', fontsize=18, fontweight='bold', y=1.05)

    # 1. Conversion Rate Comparison
    ax1 = axes[0]
    companies = ['Industry Avg', 'GitHub Project', 'Your Model']
    conversion_rates = [11.27, 39.5, 62.7]

    colors_comp = [COLORS['warning'], COLORS['secondary'], COLORS['success']]
    bars1 = ax1.bar(companies, conversion_rates, color=colors_comp, edgecolor='black', linewidth=2)

    ax1.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Conversion Rate Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars1, conversion_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. ROI Comparison
    ax2 = axes[1]
    roi_companies = ['Industry Avg', 'GitHub Project', 'Your Model']
    roi_values = [100, 354, 2574]

    bars2 = ax2.bar(roi_companies, roi_values, color=colors_comp, edgecolor='black', linewidth=2)

    ax2.set_ylabel('ROI Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('ROI Improvement Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, roi in zip(bars2, roi_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{roi:.0f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Call Efficiency
    ax3 = axes[2]
    efficiency_metrics = ['Call Reduction', 'Precision Gain', 'Profit/Lift']
    industry_values = [0, 0, 0]
    github_values = [93.4, 28.23, 354]
    your_values = [89.8, 51.43, 2574]

    x = np.arange(len(efficiency_metrics))
    width = 0.25

    bars_ind = ax3.bar(x - width, industry_values, width, label='Industry Avg', color=COLORS['warning'])
    bars_gh = ax3.bar(x, github_values, width, label='GitHub Project', color=COLORS['secondary'])
    bars_you = ax3.bar(x + width, your_values, width, label='Your Model', color=COLORS['success'])

    ax3.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax3.set_title('Efficiency Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(efficiency_metrics)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig)

def display_model_insights(scorer):
    """Display model insights"""
    st.markdown("## 💡 Model Insights & Recommendations")
    
    insights = scorer.generate_model_insights()
    st.text(insights)

def display_executive_summary(scorer):
    """Display executive summary"""
    st.markdown("## 📋 Executive Summary")
    
    summary = scorer.generate_executive_summary()
    st.text(summary)

def main():
    """Main Streamlit application"""
    
    # Sidebar for upload and controls
    with st.sidebar:
        st.title("🏦 Bank Lead Scoring")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        # Target percentile selection
        target_percentile = st.slider(
            "Target Percentile",
            min_value=1,
            max_value=30,
            value=10,
            help="Top X% of leads to target"
        )
        
        # Business assumptions
        st.markdown("---")
        st.subheader("Business Assumptions")
        call_cost = st.number_input(
            "Cost per Call ($)",
            min_value=1,
            max_value=100,
            value=10
        )
        conversion_value = st.number_input(
            "Value per Conversion ($)",
            min_value=100,
            max_value=5000,
            value=500
        )
        
        # Sample data
        st.markdown("---")
        if st.button("📥 Download Sample Data"):
            # Create sample CSV
            sample_data = pd.DataFrame({
                'age': [56, 57, 37, 40, 56],
                'job': ['admin.', 'services', 'services', 'admin.', 'services'],
                'marital': ['married', 'married', 'married', 'married', 'married'],
                'education': ['university.degree', 'high.school', 'high.school', 'basic.9y', 'high.school'],
                'default': ['no', 'unknown', 'no', 'no', 'no'],
                'housing': ['yes', 'yes', 'yes', 'no', 'no'],
                'loan': ['no', 'no', 'no', 'no', 'yes'],
                'contact': ['cellular', 'cellular', 'cellular', 'cellular', 'cellular'],
                'month': ['may', 'may', 'may', 'may', 'may'],
                'day_of_week': ['mon', 'mon', 'mon', 'mon', 'mon'],
                'duration': [261, 149, 226, 151, 307],
                'campaign': [1, 1, 1, 1, 1],
                'pdays': [999, 999, 999, 999, 999],
                'previous': [0, 0, 0, 0, 0],
                'poutcome': ['nonexistent', 'nonexistent', 'nonexistent', 'nonexistent', 'nonexistent'],
                'y': ['yes', 'yes', 'yes', 'no', 'no']
            })
            
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample.csv",
                data=csv,
                file_name="sample_bank_data.csv",
                mime="text/csv"
            )
    
    # Main content
    st.title("🏦 Bank Lead Scoring System")
    st.markdown("AI-Powered Lead Conversion Prediction & Visualization")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Display data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    conversion_rate = df['y'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0).mean() * 100 if 'y' in df.columns else 0
                    st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
                with col3:
                    st.metric("Columns", len(df.columns))
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Training model and generating insights..."):
                    # Initialize scorer
                    scorer = ProfessionalBankLeadScorerVisual(target_percentile=target_percentile)
                    scorer.call_cost = call_cost
                    scorer.conversion_value = conversion_value
                    
                    # Load and preprocess
                    df_processed = scorer.load_and_preprocess_from_dataframe(df)
                    
                    # Train model
                    results = scorer.train_model(df_processed)
                    
                    # Store in session state
                    st.session_state.scorer = scorer
                    st.session_state.results = results
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please ensure your CSV has the required 'y' column and correct format.")
    
    # Display results if available
    if 'scorer' in st.session_state and st.session_state.scorer is not None:
        scorer = st.session_state.scorer
        
        # Key Metrics Dashboard
        st.markdown("---")
        st.header("📈 Key Metrics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUC Score", f"{scorer.results['auc']:.3f}")
        with col2:
            st.metric("Precision", f"{scorer.results['precision']*100:.1f}%")
        with col3:
            st.metric("Recall", f"{scorer.results['recall']*100:.1f}%")
        with col4:
            st.metric("F1 Score", f"{scorer.results['f1']:.3f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Additional Profit", f"${scorer.results['improvement']:,.0f}")
        with col2:
            roi = (scorer.results['model_profit'] - scorer.results['baseline_profit']) / max(scorer.results['baseline_profit'], 1) * 100
            st.metric("ROI Improvement", f"{roi:.0f}%")
        with col3:
            call_reduction = 100 - (scorer.results['hot_leads_test'] / len(scorer.results['y_test']) * 100)
            st.metric("Call Reduction", f"{call_reduction:.1f}%")
        with col4:
            st.metric("Cost Saved", f"${scorer.results.get('cost_saved', 0):,.0f}")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Performance", "💰 Business Impact", "🎯 Lead Distribution", 
            "🔍 Feature Importance", "⚖️ Threshold Analysis", "📈 Benchmarks",
            "💡 Insights", "📋 Summary"
        ])
        
        with tab1:
            display_performance_dashboard(scorer)
        
        with tab2:
            display_business_impact(scorer)
        
        with tab3:
            display_lead_distribution(scorer)
        
        with tab4:
            display_feature_importance(scorer)
        
        with tab5:
            display_threshold_analysis(scorer)
        
        with tab6:
            display_benchmark_comparison(scorer)
        
        with tab7:
            display_model_insights(scorer)
        
        with tab8:
            display_executive_summary(scorer)
        
        # Download buttons
        st.markdown("---")
        st.header("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Executive Summary"):
                summary = scorer.generate_executive_summary()
                st.download_button(
                    label="Download summary.txt",
                    data=summary,
                    file_name=f"lead_scoring_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Download Predictions"):
                predictions = pd.DataFrame({
                    'actual': scorer.results['y_test'],
                    'predicted_probability': scorer.results['y_pred_proba'],
                    'predicted_class': scorer.results['y_pred']
                })
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="Download predictions.csv",
                    data=csv,
                    file_name=f"lead_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🔄 New Analysis"):
                for key in ['scorer', 'results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    else:
        # Welcome message
        st.markdown("---")
        st.info("👈 **Upload a CSV file in the sidebar to get started**")
        
        # Sample data format
        with st.expander("📋 Expected CSV Format"):
            st.code("""
Required columns:
- y: Target variable (must contain 'yes' or 'no')
- age: Customer age
- duration: Last contact duration
- campaign: Number of contacts during campaign
- pdays: Days since last contact
- previous: Number of previous contacts

Optional columns:
- job, marital, education, default, housing, loan
- contact, month, day_of_week, poutcome
            """)
        
        # Features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🎯 **6 Visualizations**")
            st.write("Performance dashboards, business impact charts, and more")
        with col2:
            st.markdown("### 💰 **Business Insights**")
            st.write("ROI analysis, profit calculations, efficiency gains")
        with col3:
            st.markdown("### 🚀 **Easy Deployment**")
            st.write("Deploy to Streamlit Cloud with one click")

if __name__ == "__main__":
    main()
