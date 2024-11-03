import streamlit as st
import torch
from pathlib import Path
import plotly.express as px
import pandas as pd
from typing import List, Dict, Tuple
import re

# Import from local file
from shakespeare_gpt import (
    ShakespeareDataset,
    EnhancedShakespeareBot,
    DEVICE,
    GPTConfig,
    GPT
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'bot' not in st.session_state:
    st.session_state.bot = None

def initialize_bot():
    """Initialize the Shakespeare bot and store in session state."""
    if st.session_state.bot is None:
        with st.spinner('Initializing Shakespeare bot...'):
            st.session_state.bot = EnhancedShakespeareBot()
            st.success('Bot initialized successfully!')

def display_character_sentiment_chart(bot):
    """Create and display a character sentiment analysis chart."""
    characters = bot.dataset.get_character_list()[:10]  # Top 10 characters
    sentiments = []
    
    for char in characters:
        sentiment = bot.analyzer.get_character_sentiment(char)
        sentiments.append({
            'Character': char,
            'Sentiment': sentiment['sentiment_ratio'],
            'Positive': sentiment['positive_count'],
            'Negative': sentiment['negative_count']
        })
    
    df = pd.DataFrame(sentiments)
    fig = px.bar(
        df,
        x='Character',
        y='Sentiment',
        title='Character Sentiment Analysis',
        color='Sentiment',
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig)

def display_character_relationships(bot):
    """Create and display a character relationships network."""
    relationships = bot.analyzer.analyze_character_relationships()
    
    # Convert relationships to a format suitable for visualization
    nodes = list(set([char for pair in relationships.keys() for char in pair]))
    edges = [{'source': pair[0], 'target': pair[1], 'weight': count} 
             for pair, count in relationships.items()]
    
    # Create a DataFrame for the relationships
    df = pd.DataFrame(edges)
    df = df.sort_values('weight', ascending=False).head(20)  # Show top 20 relationships
    
    fig = px.bar(
        df,
        x='source',
        y='weight',
        color='target',
        title='Top Character Interactions',
        labels={'source': 'Character 1', 'target': 'Character 2', 'weight': 'Scenes Together'}
    )
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Shakespeare Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .bot-message {
            background-color: #fff;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and introduction
    st.title("🎭 Shakespeare Interactive Chatbot")
    st.markdown("""
        Welcome to the Shakespeare Interactive Chatbot! Explore the works of William Shakespeare
        through conversation, analysis, and visualization. Ask questions about characters,
        search for quotes, or analyze relationships and sentiments.
    """)
    
    # Initialize bot
    initialize_bot()
    
    # Sidebar with options
    st.sidebar.title("Analytics Dashboard")
    analysis_option = st.sidebar.selectbox(
        "Choose Analysis View",
        ["Chat", "Play Statistics"]
    )
    
    if analysis_option == "Chat":
        # Main chat interface
        st.markdown("### Chat with Shakespeare Bot")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">You: {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">Bot: {message["content"]}</div>', 
                          unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Your message:", key="user_input")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get bot response
            response = st.session_state.bot.process_query(user_input)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat display
            st.experimental_rerun()
            
    elif analysis_option == "Play Statistics":
        st.markdown("### Play Statistics")
        stats = st.session_state.bot.analyzer.get_play_statistics()
        
        # Display statistics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Words", f"{stats['total_words']:,}")
            st.metric("Total Characters", stats['total_characters'])
            st.metric("Unique Words", f"{stats['unique_words']:,}")
            
        with col2:
            st.metric("Total Sentences", f"{stats['total_sentences']:,}")
            st.metric("Average Sentence Length", f"{stats['average_sentence_length']:.1f}")
            st.metric("Most Talkative Character", 
                     f"{stats['most_talkative_character'][0]}: {stats['most_talkative_character'][1]:,} words")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        Made with ❤️ by Samuel using Streamlit. This chatbot uses advanced NLP to help you explore
        Shakespeare's works. Try asking about characters, searching for quotes, or analyzing
        relationships!
    """)

if __name__ == "__main__":
    main()
