"""
Chat Interface Component for AI Trading Assistant
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime
import time

class ChatInterface:
    """Component for AI trading assistant chat interface"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.max_history = 50
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ""
    
    def render(self):
        """Render the chat interface"""
        st.markdown("### AI Trading Assistant")
        st.markdown("Ask me anything about trading, market analysis, or portfolio management!")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            self._render_chat_history()
        
        # Input area
        self._render_input_area()
        
        # Quick action buttons
        self._render_quick_actions()
        
        # Chat settings
        self._render_chat_settings()
    
    def _render_chat_history(self):
        """Render chat history"""
        if not st.session_state.chat_history:
            # Welcome message
            st.markdown("""
            <div style="
                background: #e3f2fd;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #2196f3;
            ">
                <strong>🤖 AI Assistant:</strong> Hello! I'm your AI trading assistant. I can help you with:
                <ul>
                    <li>Market analysis and insights</li>
                    <li>Portfolio performance review</li>
                    <li>Trading strategy discussions</li>
                    <li>Risk management advice</li>
                    <li>Technical analysis questions</li>
                </ul>
                What would you like to know?
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                self._render_user_message(message)
            else:
                self._render_assistant_message(message)
    
    def _render_user_message(self, message: Dict):
        """Render user message"""
        timestamp = message.get('timestamp', datetime.now()).strftime('%H:%M')
        
        st.markdown(f"""
        <div style="
            background: #f5f5f5;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            margin-left: 2rem;
            border-left: 4px solid #ff9800;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>👤 You:</strong>
                <small style="color: #666;">{timestamp}</small>
            </div>
            <div style="margin-top: 0.5rem;">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_assistant_message(self, message: Dict):
        """Render assistant message"""
        timestamp = message.get('timestamp', datetime.now()).strftime('%H:%M')
        
        st.markdown(f"""
        <div style="
            background: #e8f5e8;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            margin-right: 2rem;
            border-left: 4px solid #4caf50;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>🤖 AI Assistant:</strong>
                <small style="color: #666;">{timestamp}</small>
            </div>
            <div style="margin-top: 0.5rem;">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_input_area(self):
        """Render chat input area"""
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Type your message...",
                    value="",
                    placeholder="Ask about market conditions, portfolio, or trading strategies...",
                    key="chat_input_field"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send 📤")
            
            if submit_button and user_input:
                self._handle_user_input(user_input)
    
    def _render_quick_actions(self):
        """Render quick action buttons"""
        st.markdown("#### Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Market Summary"):
                self._handle_quick_action("What's the current market summary?")
        
        with col2:
            if st.button("💼 Portfolio Status"):
                self._handle_quick_action("Show me my portfolio status")
        
        with col3:
            if st.button("🎯 Trading Opportunities"):
                self._handle_quick_action("What are the best trading opportunities right now?")
        
        with col4:
            if st.button("⚠️ Risk Analysis"):
                self._handle_quick_action("Analyze my current risk exposure")
        
        # Additional quick actions
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            if st.button("📈 Technical Analysis"):
                self._handle_quick_action("Provide technical analysis for my top positions")
        
        with col6:
            if st.button("💡 Strategy Suggestions"):
                self._handle_quick_action("Suggest trading strategies for current market conditions")
        
        with col7:
            if st.button("📰 Market News"):
                self._handle_quick_action("What's the latest market news I should know about?")
        
        with col8:
            if st.button("🔄 Rebalancing"):
                self._handle_quick_action("Should I rebalance my portfolio?")
    
    def _render_chat_settings(self):
        """Render chat settings"""
        with st.expander("Chat Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                # AI model selection
                ai_model = st.selectbox(
                    "AI Model",
                    ["claude-sonnet-4-20250514", "gpt-4o"],
                    index=0
                )
                
                # Response length
                response_length = st.selectbox(
                    "Response Length",
                    ["Short", "Medium", "Detailed"],
                    index=1
                )
            
            with col2:
                # Context awareness
                context_aware = st.checkbox("Context Aware", value=True)
                
                # Include portfolio data
                include_portfolio = st.checkbox("Include Portfolio Data", value=True)
            
            # Clear chat history
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col2:
                if st.button("Export Chat"):
                    self._export_chat_history()
    
    def _handle_user_input(self, user_input: str):
        """Handle user input"""
        # Add user message to history
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(user_message)
        
        # Show thinking indicator
        with st.spinner("🤖 AI is thinking..."):
            # Get AI response
            ai_response = self._get_ai_response(user_input)
        
        # Add AI response to history
        ai_message = {
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(ai_message)
        
        # Trim history if too long
        if len(st.session_state.chat_history) > self.max_history:
            st.session_state.chat_history = st.session_state.chat_history[-self.max_history:]
        
        # Rerun to update display
        st.rerun()
    
    def _handle_quick_action(self, message: str):
        """Handle quick action button press"""
        self._handle_user_input(message)
    
    def _get_ai_response(self, message: str) -> str:
        """Get AI response from backend"""
        try:
            # Call backend API
            response = requests.post(
                f"{self.api_base_url}/api/chat",
                json={"message": message},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', 'I apologize, but I received an empty response.')
            else:
                return f"I'm sorry, I encountered an error (Status: {response.status_code}). Please try again."
        
        except requests.exceptions.Timeout:
            return "I'm taking longer than usual to respond. Please try again or ask a simpler question."
        
        except requests.exceptions.ConnectionError:
            return "I'm unable to connect to the AI service right now. Please check your connection and try again."
        
        except Exception as e:
            return f"I encountered an unexpected error: {str(e)}. Please try again."
    
    def _export_chat_history(self):
        """Export chat history"""
        try:
            # Convert chat history to JSON
            export_data = {
                'export_date': datetime.now().isoformat(),
                'chat_history': [
                    {
                        'role': msg['role'],
                        'content': msg['content'],
                        'timestamp': msg['timestamp'].isoformat()
                    }
                    for msg in st.session_state.chat_history
                ]
            }
            
            # Create download button
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="Download Chat History",
                data=json_string,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error exporting chat history: {str(e)}")
    
    def add_system_message(self, message: str):
        """Add system message to chat"""
        system_message = {
            'role': 'system',
            'content': message,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(system_message)
    
    def get_chat_context(self) -> List[Dict]:
        """Get recent chat context for AI"""
        # Return last 10 messages for context
        return st.session_state.chat_history[-10:]
    
    def clear_chat(self):
        """Clear chat history"""
        st.session_state.chat_history = []
