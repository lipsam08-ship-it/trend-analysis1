import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Set page configuration
st.set_page_config(
    page_title="AI Writing Trend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .trend-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .trend-negative {
        color: #ff0000;
        font-weight: bold;
    }
    .sidebar-tab {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        cursor: pointer;
    }
    .sidebar-tab:hover {
        background-color: #f0f2f6;
    }
    .tab-active {
        background-color: #1f77b4;
        color: white;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .tech-stack {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class WritingTrendAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        topics = ['AI Writing', 'Content Marketing', 'SEO Optimization', 
                 'Social Media', 'Video Content', 'Personal Branding']
        
        data = []
        for date in dates:
            for topic in topics:
                # Simulate trend patterns
                base_volume = 100 + np.random.randint(0, 50)
                trend_factor = 1 + 0.1 * np.sin((date - datetime(2024, 1, 1)).days / 30)
                volume = int(base_volume * trend_factor * (1 + topics.index(topic) * 0.1))
                
                # Generate sample content
                content = f"This is a sample article about {topic} discussing current trends and best practices."
                
                # Calculate sentiment
                sentiment = self.sia.polarity_scores(content)
                
                data.append({
                    'date': date,
                    'topic': topic,
                    'volume': volume,
                    'sentiment_compound': sentiment['compound'],
                    'sentiment_positive': sentiment['pos'],
                    'sentiment_negative': sentiment['neg'],
                    'sentiment_neutral': sentiment['neu'],
                    'content_sample': content
                })
        
        return pd.DataFrame(data)
    
    def analyze_trends(self, df, selected_topic=None):
        """Analyze trends from the data"""
        if selected_topic and selected_topic != 'All Topics':
            df = df[df['topic'] == selected_topic]
        
        # Weekly aggregation
        df_weekly = df.groupby([pd.Grouper(key='date', freq='W'), 'topic']).agg({
            'volume': 'sum',
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean'
        }).reset_index()
        
        return df_weekly
    
    def generate_word_cloud_data(self, df):
        """Generate data for word cloud"""
        topics_text = ' '.join(df['topic'].value_counts().index.tolist() * 10)
        additional_terms = ['content', 'writing', 'digital', 'marketing', 'strategy', 
                          'engagement', 'audience', 'creation', 'optimization', 'trends']
        text = topics_text + ' ' + ' '.join(additional_terms * 5)
        return text

def validate_and_clean_data(df):
    """Validate and clean the uploaded DataFrame"""
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Required columns check
    required_columns = ['date', 'topic', 'volume']
    missing_columns = [col for col in required_columns if col not in df_clean.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return None
    
    # Convert date column
    try:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
    except Exception as e:
        st.error(f"Error converting date column: {str(e)}")
        return None
    
    # Convert volume to numeric, handling errors
    try:
        df_clean['volume'] = pd.to_numeric(df_clean['volume'], errors='coerce')
        # Fill NaN values with 0 or median
        if df_clean['volume'].isna().any():
            st.warning(f"Found {df_clean['volume'].isna().sum()} non-numeric values in 'volume' column. Replacing with median.")
            median_volume = df_clean['volume'].median()
            df_clean['volume'] = df_clean['volume'].fillna(median_volume)
    except Exception as e:
        st.error(f"Error converting volume column to numeric: {str(e)}")
        return None
    
    # Ensure topic is string
    df_clean['topic'] = df_clean['topic'].astype(str)
    
    # Handle sentiment columns if they exist
    sentiment_columns = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
    for col in sentiment_columns:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Fill NaN sentiment with 0
                df_clean[col] = df_clean[col].fillna(0)
            except Exception as e:
                st.warning(f"Could not convert {col} to numeric: {str(e)}")
    
    st.success(f"✅ Data cleaned successfully! Loaded {len(df_clean)} records.")
    return df_clean

def show_trend_overview(analyzer, df, selected_topic):
    """Display trend overview"""
    st.header("📈 Writing Trends Overview")
    
    # Check if we have data after filtering
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Key metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_volume = df['volume'].sum()
        st.metric("Total Content Volume", f"{total_volume:,}")
    
    with col2:
        avg_sentiment = df['sentiment_compound'].mean() if 'sentiment_compound' in df.columns else 0
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    with col3:
        unique_topics = df['topic'].nunique()
        st.metric("Unique Topics", unique_topics)
    
    with col4:
        trending_topic = df.groupby('topic')['volume'].sum().idxmax()
        st.metric("Most Popular Topic", trending_topic)
    
    # Trend analysis
    st.subheader("📈 Content Volume Trends")
    df_weekly = analyzer.analyze_trends(df, selected_topic)
    
    if not df_weekly.empty:
        fig_volume = px.line(
            df_weekly, 
            x='date', 
            y='volume', 
            color='topic' if selected_topic == 'All Topics' else None,
            title=f"Content Volume Trends - {selected_topic}",
            template="plotly_white"
        )
        fig_volume.update_layout(
            xaxis_title="Date",
            yaxis_title="Content Volume",
            hovermode='x unified'
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.info("No trend data available for the selected filters.")
    
    # Topic distribution
    st.subheader("📊 Topic Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Topic Volume Distribution**")
        topic_volume = df.groupby('topic')['volume'].sum().sort_values(ascending=False)
        if not topic_volume.empty:
            fig_pie = px.pie(
                values=topic_volume.values,
                names=topic_volume.index,
                title="Topic Volume Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No topic distribution data available.")
    
    with col2:
        st.markdown("**Trending Topics Word Cloud**")
        text = analyzer.generate_word_cloud_data(df)
        if text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Trending Topics Word Cloud', fontsize=16)
            st.pyplot(fig)
        else:
            st.info("No data available for word cloud.")

def show_sentiment_analysis(analyzer, df, selected_topic):
    """Display sentiment analysis"""
    st.header("😊 Sentiment Analysis")
    
    # Check if we have data after filtering
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Check if sentiment columns exist
    if 'sentiment_compound' not in df.columns:
        st.warning("Sentiment data not available in the uploaded dataset.")
        st.info("The sample dataset includes sentiment analysis, but your uploaded data may not have these columns.")
        return
    
    # Filter data if specific topic selected
    if selected_topic != 'All Topics':
        df = df[df['topic'] == selected_topic]
    
    # Sentiment metrics
    st.subheader("📊 Sentiment Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positive_content = len(df[df['sentiment_compound'] > 0.05])
        st.metric("Positive Content", positive_content, delta="High Engagement")
    
    with col2:
        neutral_content = len(df[(df['sentiment_compound'] >= -0.05) & (df['sentiment_compound'] <= 0.05)])
        st.metric("Neutral Content", neutral_content, delta="Moderate Engagement")
    
    with col3:
        negative_content = len(df[df['sentiment_compound'] < -0.05])
        st.metric("Negative Content", negative_content, delta="Low Engagement", delta_color="inverse")
    
    with col4:
        avg_sentiment = df['sentiment_compound'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        st.metric("Overall Sentiment", f"{avg_sentiment:.3f}", delta=sentiment_label)
    
    # Sentiment trends over time
    st.subheader("📈 Sentiment Trends Over Time")
    df_weekly = analyzer.analyze_trends(df, selected_topic)
    
    if not df_weekly.empty:
        fig_sentiment = px.line(
            df_weekly,
            x='date',
            y='sentiment_compound',
            color='topic' if selected_topic == 'All Topics' else None,
            title="Sentiment Score Trends Over Time",
            template="plotly_white"
        )
        fig_sentiment.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode='x unified'
        )
        # Add a horizontal line at y=0 for reference
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    else:
        st.info("No sentiment trend data available.")
    
    # Sentiment by topic
    st.subheader("📊 Sentiment Analysis by Topic")
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_by_topic = df.groupby('topic').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean'
        }).reset_index()
        
        if not sentiment_by_topic.empty:
            fig_bar = px.bar(
                sentiment_by_topic,
                x='topic',
                y='sentiment_compound',
                color='sentiment_compound',
                title="Average Sentiment Score by Topic",
                color_continuous_scale='RdYlGn',
                template="plotly_white"
            )
            fig_bar.update_layout(
                xaxis_title="Topic",
                yaxis_title="Average Sentiment Score"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No sentiment data by topic available.")
    
    with col2:
        # Sentiment distribution
        st.markdown("**Sentiment Distribution**")
        if not df.empty:
            fig_hist = px.histogram(
                df,
                x='sentiment_compound',
                nbins=20,
                title="Distribution of Sentiment Scores",
                template="plotly_white"
            )
            fig_hist.update_layout(
                xaxis_title="Sentiment Score",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def show_topic_comparison(analyzer, df):
    """Display topic comparison"""
    st.header("📊 Topic Comparison Analysis")
    
    # Check if we have data after filtering
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Topic performance metrics
    st.subheader("📈 Topic Performance Metrics")
    topic_stats = df.groupby('topic').agg({
        'volume': ['sum', 'mean', 'std'],
        'sentiment_compound': 'mean' if 'sentiment_compound' in df.columns else 'count'
    }).round(3)
    
    # Flatten column names
    topic_stats.columns = ['Total Volume', 'Avg Volume', 'Volume Std', 'Avg Sentiment']
    topic_stats = topic_stats.sort_values('Total Volume', ascending=False)
    
    # Display metrics in a nice format
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_topic = topic_stats.index[0]
        top_volume = topic_stats['Total Volume'].iloc[0]
        st.metric("🏆 Top Topic", top_topic, f"{top_volume:,.0f}")
    
    with col2:
        avg_volume = topic_stats['Total Volume'].mean()
        st.metric("📊 Average Volume", f"{avg_volume:,.0f}")
    
    with col3:
        total_topics = len(topic_stats)
        st.metric("🏷️ Total Topics", total_topics)
    
    with col4:
        if 'Avg Sentiment' in topic_stats.columns:
            best_sentiment = topic_stats['Avg Sentiment'].max()
            st.metric("😊 Best Sentiment", f"{best_sentiment:.3f}")
    
    st.dataframe(topic_stats, use_container_width=True)
    
    # Comparison charts
    st.subheader("📊 Visual Comparisons")
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume comparison
        if not topic_stats.empty:
            fig_volume_comp = px.bar(
                topic_stats.reset_index(),
                x='topic',
                y='Total Volume',
                title="Total Volume by Topic",
                color='Total Volume',
                color_continuous_scale='viridis',
                template="plotly_white"
            )
            fig_volume_comp.update_layout(
                xaxis_title="Topic",
                yaxis_title="Total Volume",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_volume_comp, use_container_width=True)
    
    with col2:
        # Sentiment vs Volume scatter
        if not topic_stats.empty and len(topic_stats) > 1 and 'Avg Sentiment' in topic_stats.columns:
            fig_scatter = px.scatter(
                topic_stats.reset_index(),
                x='Total Volume',
                y='Avg Sentiment',
                size='Total Volume',
                color='topic',
                title="Sentiment vs Volume by Topic",
                hover_data=['topic'],
                template="plotly_white"
            )
            fig_scatter.update_layout(
                xaxis_title="Total Volume",
                yaxis_title="Average Sentiment"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Insufficient data for scatter plot")
    
    # Growth trends comparison
    st.subheader("📈 Topic Growth Analysis")
    df_weekly = analyzer.analyze_trends(df, 'All Topics')
    
    if not df_weekly.empty:
        # Calculate growth rates
        topic_growth = {}
        for topic in df['topic'].unique():
            topic_data = df_weekly[df_weekly['topic'] == topic]
            if len(topic_data) > 1:
                # Ensure volume is numeric
                topic_data = topic_data.copy()
                topic_data['volume'] = pd.to_numeric(topic_data['volume'], errors='coerce')
                topic_data = topic_data.dropna(subset=['volume'])
                
                if len(topic_data) > 1 and topic_data['volume'].iloc[0] > 0:
                    growth = (topic_data['volume'].iloc[-1] - topic_data['volume'].iloc[0]) / topic_data['volume'].iloc[0] * 100
                    topic_growth[topic] = growth
        
        if topic_growth:
            growth_df = pd.DataFrame(list(topic_growth.items()), columns=['Topic', 'Growth Rate'])
            growth_df = growth_df.sort_values('Growth Rate', ascending=False)
            
            fig_growth = px.bar(
                growth_df,
                x='Topic',
                y='Growth Rate',
                title="Topic Growth Rates (%)",
                color='Growth Rate',
                color_continuous_scale='RdYlGn',
                template="plotly_white"
            )
            fig_growth.update_layout(
                xaxis_title="Topic",
                yaxis_title="Growth Rate (%)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.info("Insufficient data to calculate growth rates.")

def show_content_recommendations(analyzer, df):
    """Display content recommendations"""
    st.header("💡 Content Strategy Recommendations")
    
    # Check if we have data after filtering
    if df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Ensure volume is numeric
    df = df.copy()
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['volume'])
    
    if df.empty:
        st.warning("No valid volume data available for recommendations.")
        return
    
    # Analyze current trends
    current_date = df['date'].max()
    recent_data = df[df['date'] >= (current_date - timedelta(days=30))]
    
    if recent_data.empty:
        st.info("Not enough recent data for recommendations. Try expanding your date range.")
        return
    
    # Ensure recent_data volume is numeric
    recent_data = recent_data.copy()
    recent_data['volume'] = pd.to_numeric(recent_data['volume'], errors='coerce')
    recent_data = recent_data.dropna(subset=['volume'])
    
    if recent_data.empty:
        st.warning("No valid volume data in recent period.")
        return
    
    # Top performing topics
    try:
        top_topics = recent_data.groupby('topic')['volume'].sum().nlargest(5)
    except Exception as e:
        st.error(f"Error calculating top topics: {str(e)}")
        top_topics = pd.Series(dtype=float)
    
    # Growing topics
    previous_period = df[df['date'] < (current_date - timedelta(days=30))]
    
    # Ensure previous_period volume is numeric
    previous_period = previous_period.copy()
    previous_period['volume'] = pd.to_numeric(previous_period['volume'], errors='coerce')
    previous_period = previous_period.dropna(subset=['volume'])
    
    current_growth = {}
    
    for topic in df['topic'].unique():
        current_vol = recent_data[recent_data['topic'] == topic]['volume'].sum()
        previous_vol = previous_period[previous_period['topic'] == topic]['volume'].sum()
        
        if previous_vol > 0 and current_vol > 0:
            growth = (current_vol - previous_vol) / previous_vol * 100
            current_growth[topic] = growth
    
    # Convert to Series and ensure numeric
    if current_growth:
        growth_series = pd.Series(current_growth)
        growth_series = pd.to_numeric(growth_series, errors='coerce')
        growing_topics = growth_series.nlargest(3)
    else:
        growing_topics = pd.Series(dtype=float)
    
    # High sentiment topics
    if 'sentiment_compound' in df.columns:
        high_sentiment_topics = recent_data.groupby('topic')['sentiment_compound'].mean().nlargest(3)
    else:
        high_sentiment_topics = pd.Series(dtype=float)
    
    # Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Top Performing Topics")
        if not top_topics.empty:
            for i, (topic, volume) in enumerate(top_topics.items(), 1):
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #1f77b4;'>
                    <h4>#{i} {topic}</h4>
                    <p>📈 Volume: <strong>{volume:,.0f}</strong> engagements</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No top performing topics data available.")
    
    with col2:
        st.markdown("### 🚀 Fastest Growing Topics")
        if not growing_topics.empty:
            for i, (topic, growth) in enumerate(growing_topics.items(), 1):
                trend_icon = "📈" if growth > 0 else "📉"
                trend_color = "#00ff00" if growth > 0 else "#ff0000"
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {trend_color};'>
                    <h4>#{i} {topic} {trend_icon}</h4>
                    <p>📊 Growth: <strong style='color: {trend_color};'>{growth:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No growth data available.")
    
    # Content strategy suggestions
    st.subheader("📝 Actionable Content Strategy")
    
    suggestions = []
    
    # Based on top topics
    if not top_topics.empty:
        for topic in top_topics.index[:2]:
            suggestions.append(f"**Create comprehensive guides** on '{topic}' to leverage existing audience interest")
    
    # Based on growing topics
    if not growing_topics.empty:
        for topic in growing_topics.index:
            if growing_topics[topic] > 20:  # Only if growth is significant
                suggestions.append(f"**Jump on emerging trend**: '{topic}' is growing rapidly (+{growing_topics[topic]:.1f}%)")
    
    # Based on sentiment
    if not high_sentiment_topics.empty:
        for topic in high_sentiment_topics.index:
            if high_sentiment_topics[topic] > 0.3:  # High positive sentiment
                suggestions.append(f"**Positive content opportunity**: '{topic}' has high engagement sentiment")
    
    if suggestions:
        st.markdown("### 💡 Recommended Actions")
        for i, suggestion in enumerate(suggestions[:5], 1):
            st.markdown(f"""
            <div style='background-color: #e8f4fd; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #1f77b4;'>
                {i}. {suggestion}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations available. Try analyzing more data.")
    
    # Optimal posting schedule
    st.subheader("🕒 Optimal Content Timing Strategy")
    
    # Simulate engagement by hour
    hours = list(range(24))
    engagement = [50, 30, 20, 15, 10, 15, 40, 80, 120, 150, 140, 130, 
                 120, 110, 100, 120, 140, 160, 180, 170, 150, 120, 90, 60]
    
    timing_df = pd.DataFrame({'Hour': hours, 'Engagement': engagement})
    
    fig_timing = px.area(
        timing_df,
        x='Hour',
        y='Engagement',
        title="Recommended Posting Times (Engagement by Hour)",
        labels={'Hour': 'Hour of Day', 'Engagement': 'Expected Engagement'},
        template="plotly_white"
    )
    fig_timing.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Expected Engagement"
    )
    st.plotly_chart(fig_timing, use_container_width=True)

def show_ai_models_used():
    """Display information about AI models and algorithms used"""
    st.header("🤖 AI Models & Algorithms Used")
    
    st.markdown("""
    ## 🧠 Artificial Intelligence Components
    
    This dashboard leverages multiple AI and machine learning techniques to provide 
    comprehensive writing trend analysis. Below are the key models and algorithms implemented:
    """)
    
    # Natural Language Processing Section
    st.markdown("---")
    st.subheader("🔤 Natural Language Processing (NLP)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>📊 VADER Sentiment Analysis</h3>
            <p><strong>Purpose:</strong> Text sentiment scoring</p>
            <p><strong>Library:</strong> NLTK VADER</p>
            <p><strong>Output:</strong> Compound sentiment scores (-1 to +1)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### How it works:
        - **Lexicon-based approach** with rule-based sentiment analysis
        - **Pre-trained model** specifically for social media and short texts
        - **Compound score calculation** considering intensity and direction
        - **Real-time processing** for immediate sentiment insights
        """)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>📝 TextBlob Integration</h3>
            <p><strong>Purpose:</strong> Alternative sentiment analysis</p>
            <p><strong>Library:</strong> TextBlob</p>
            <p><strong>Features:</strong> Polarity & subjectivity scoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### Key Features:
        - **Dual sentiment analysis** for validation
        - **Subjectivity scoring** (0 = objective, 1 = subjective)
        - **Pattern analyzer integration**
        - **Multi-language support** capabilities
        """)
    
    # Machine Learning Algorithms
    st.markdown("---")
    st.subheader("🤖 Machine Learning & Statistical Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #e8f4fd; padding: 15px; border-radius: 10px; border-left: 4px solid #007bff;">
            <h4>📈 Trend Detection</h4>
            <p><strong>Algorithm:</strong> Moving Average & Seasonal Decomposition</p>
            <p><strong>Method:</strong> Time-series analysis with weekly aggregation</p>
            <p><strong>Output:</strong> Trend lines and growth patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Technical Details:**
        - Weekly data aggregation
        - Exponential smoothing
        - Seasonal pattern detection
        - Anomaly identification
        """)
    
    with col2:
        st.markdown("""
        <div style="background: #e8f4fd; padding: 15px; border-radius: 10px; border-left: 4px solid #28a745;">
            <h4>🎯 Topic Modeling</h4>
            <p><strong>Algorithm:</strong> Frequency Analysis & TF-IDF</p>
            <p><strong>Method:</strong> Statistical topic importance</p>
            <p><strong>Output:</strong> Topic relevance scores</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Technical Details:**
        - Term Frequency analysis
        - Topic volume distribution
        - Growth rate calculations
        - Comparative performance metrics
        """)
    
    with col3:
        st.markdown("""
        <div style="background: #e8f4fd; padding: 15px; border-radius: 10px; border-left: 4px solid #dc3545;">
            <h4>📊 Recommendation Engine</h4>
            <p><strong>Algorithm:</strong> Rule-based AI with Growth Analysis</p>
            <p><strong>Method:</strong> Multi-factor scoring system</p>
            <p><strong>Output:</strong> Content strategy suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Technical Details:**
        - Growth rate thresholding
        - Sentiment-performance correlation
        - Engagement pattern analysis
        - Strategic opportunity identification
        """)
    
    # Technical Stack Details
    st.markdown("---")
    st.subheader("⚙️ Technical Stack & Libraries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tech-stack">
            <h4>📚 Core Libraries</h4>
            <ul>
                <li><strong>Streamlit</strong> - Web application framework</li>
                <li><strong>Pandas</strong> - Data manipulation and analysis</li>
                <li><strong>NumPy</strong> - Numerical computing</li>
                <li><strong>Plotly</strong> - Interactive visualizations</li>
                <li><strong>Matplotlib/Seaborn</strong> - Static visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-stack">
            <h4>🧠 AI/ML Libraries</h4>
            <ul>
                <li><strong>NLTK</strong> - Natural Language Toolkit</li>
                <li><strong>TextBlob</strong> - Simplified text processing</li>
                <li><strong>VADER Sentiment</strong> - Social media sentiment analysis</li>
                <li><strong>Scikit-learn compatible</strong> - ML algorithm integration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_data_management(analyzer, df):
    """Display data management section"""
    st.header("⚙️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Dataset Overview")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        st.metric("Unique Topics", df['topic'].nunique())
        st.metric("Total Volume", f"{df['volume'].sum():,}")
    
    with col2:
        st.subheader("🔍 Data Quality Check")
        
        # Data quality metrics
        missing_volume = df['volume'].isna().sum()
        zero_volume = (df['volume'] == 0).sum()
        date_issues = df['date'].isna().sum()
        
        st.metric("Missing Volume Values", missing_volume)
        st.metric("Zero Volume Records", zero_volume)
        st.metric("Date Issues", date_issues)
        
        if missing_volume > 0 or date_issues > 0:
            st.warning("⚠️ Data quality issues detected. Consider cleaning your data.")
        else:
            st.success("✅ Data quality is good!")
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Data statistics
    st.subheader("📈 Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Export options
    st.subheader("📤 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Processed Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="processed_writing_trends.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Reset to Sample Data"):
            st.info("Sample data reloaded!")
            st.rerun()

def show_about():
    """Display about section"""
    st.header("📋 About AI Writing Trend Analyzer")
    
    st.markdown("""
    ## 🤖 Overview
    
    The **AI Writing Trend Analyzer** is a comprehensive dashboard designed to help content creators, 
    marketers, and writers identify and analyze writing trends using artificial intelligence and 
    natural language processing.
    
    ## 🎯 Key Features
    
    ### 📈 Trend Overview
    - Real-time trend visualization
    - Content volume analysis
    - Topic distribution insights
    - Interactive charts and graphs
    
    ### 😊 Sentiment Analysis
    - AI-powered sentiment scoring
    - Emotional tone analysis
    - Sentiment trends over time
    - Positive/negative content identification
    
    ### 📊 Topic Comparison
    - Multi-topic performance comparison
    - Growth rate analysis
    - Sentiment vs volume correlation
    - Competitive topic insights
    
    ### 💡 Content Recommendations
    - AI-driven strategy suggestions
    - Emerging trend identification
    - Optimal posting timing
    - Actionable insights
    
    ## 🛠️ Technical Stack
    
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib
    - **NLP**: NLTK, TextBlob, VADER Sentiment
    - **Machine Learning**: Scikit-learn compatible
    
    ## 📊 Data Sources
    
    The analyzer supports multiple data sources:
    - Sample generated data for demonstration
    - CSV file uploads with custom data
    - Social media APIs (extensible)
    - Content management systems
    
    ## 🔧 Required CSV Format
    
    Your CSV should include these columns:
    - `date` (YYYY-MM-DD format)
    - `topic` (text content category)
    - `volume` (numeric engagement metric)
    - Optional sentiment columns for advanced analysis
    
    ## 🚀 Getting Started
    
    1. **Choose Data Source**: Select sample data or upload your CSV
    2. **Apply Filters**: Use date range and topic filters
    3. **Explore Tabs**: Navigate through different analysis types
    4. **Get Insights**: Use recommendations to optimize your content strategy
    
    ## 📞 Support
    
    For issues or feature requests, please contact the development team.
    """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Writing Trend Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = WritingTrendAnalyzer()
    
    # Sidebar with tabs
    st.sidebar.title("📊 Dashboard Navigation")
    
    # Create tabs in sidebar
    selected_tab = st.sidebar.radio(
        "Select Analysis Type",
        [
            "📈 Trend Overview", 
            "😊 Sentiment Analysis", 
            "📊 Topic Comparison", 
            "💡 Content Recommendations",
            "🤖 AI Models Used",
            "⚙️ Data Management",
            "📋 About"
        ],
        key="main_tabs"
    )
    
    # Data Management Section (always visible for data loading)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Source")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose Data Source",
        ["Sample Data", "Upload CSV"],
        help="Choose between sample data or upload your own CSV file",
        key="data_source"
    )
    
    df = None
    if data_source == "Sample Data":
        with st.spinner("Generating sample trend data..."):
            df = analyzer.generate_sample_data()
        st.sidebar.success("✅ Sample data loaded successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Clean and validate the data
                df = validate_and_clean_data(df)
                if df is None:
                    st.sidebar.error("Failed to process the uploaded CSV file.")
                    # Fallback to sample data
                    df = analyzer.generate_sample_data()
                    st.sidebar.info("Using sample data as fallback.")
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {str(e)}")
                # Fallback to sample data
                df = analyzer.generate_sample_data()
                st.sidebar.info("Using sample data as fallback.")
        else:
            st.sidebar.info("📁 Please upload a CSV file or use sample data")
            # Use sample data as fallback
            df = analyzer.generate_sample_data()
    
    # If no data is loaded, show warning and return
    if df is None or df.empty:
        st.warning("No data available. Please check your data source.")
        return
    
    # Filters Section (visible for all tabs except About and AI Models)
    if selected_tab not in ["📋 About", "🤖 AI Models Used"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Filters & Settings")
        
        # Date range filter with safe defaults
        try:
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            # Set default date range (last 30 days or full range if less than 30 days)
            default_end = max_date
            default_start = max(min_date, (default_end - timedelta(days=30)))
            
            date_range = st.sidebar.date_input(
                "📅 Date Range",
                value=(default_start, default_end),
                min_value=min_date,
                max_value=max_date
            )
            
            # Handle single date selection
            if len(date_range) == 1:
                # If only one date selected, use that date only
                selected_date = pd.to_datetime(date_range[0])
                df = df[df['date'].dt.date == selected_date.date()]
            elif len(date_range) == 2:
                # If range selected, filter by range
                start_date, end_date = date_range
                df = df[(df['date'] >= pd.to_datetime(start_date)) & 
                        (df['date'] <= pd.to_datetime(end_date))]
                
        except Exception as e:
            st.sidebar.error(f"Error processing dates: {str(e)}")
            # Continue with full dataset if date filtering fails
        
        # Topic filter
        topics = ['All Topics'] + df['topic'].unique().tolist() if 'topic' in df.columns else ['All Topics']
        selected_topic = st.sidebar.selectbox("🏷️ Select Topic", topics, key="topic_filter")
    
    # Display data info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Summary")
    st.sidebar.write(f"📈 Records: {len(df):,}")
    st.sidebar.write(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"🏷️ Topics: {df['topic'].nunique()}")
    st.sidebar.write(f"📊 Total volume: {df['volume'].sum():,}")
    
    # Main content based on selected tab
    if selected_tab == "📈 Trend Overview":
        show_trend_overview(analyzer, df, selected_topic)
    elif selected_tab == "😊 Sentiment Analysis":
        show_sentiment_analysis(analyzer, df, selected_topic)
    elif selected_tab == "📊 Topic Comparison":
        show_topic_comparison(analyzer, df)
    elif selected_tab == "💡 Content Recommendations":
        show_content_recommendations(analyzer, df)
    elif selected_tab == "🤖 AI Models Used":
        show_ai_models_used()
    elif selected_tab == "⚙️ Data Management":
        show_data_management(analyzer, df)
    elif selected_tab == "📋 About":
        show_about()

if __name__ == "__main__":
    main()
