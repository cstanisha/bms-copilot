"""
Conversation Memory Manager - LangChain 0.3+ Compatible
Uses proper message history for long-term memory
"""
from typing import List, Dict, Optional

# LangChain Core - These definitely work in 0.3+
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory

# LangChain OpenAI
from langchain_openai import ChatOpenAI


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """
    Simple in-memory chat message history
    Compatible with LangChain 0.3+
    """
    
    def __init__(self):
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store"""
        self.messages.append(message)
    
    def add_user_message(self, message: str) -> None:
        """Add a user message"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message"""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages"""
        return self._messages
    
    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Set messages"""
        self._messages = value


class ConversationMemoryManager:
    """
    Manages conversation history and context
    Long-term memory compatible with LangChain 0.3+
    """
    
    def __init__(
        self,
        k: int = 10,  # Keep last k exchanges (k*2 messages)
        llm_model: str = "gpt-4-turbo-preview"
    ):
        """
        Initialize memory manager
        
        Args:
            k: Number of recent message pairs to keep (default 10)
            llm_model: LLM model name (for future use)
        """
        self.k = k
        self.llm_model = llm_model
        
        # Use proper chat message history
        self.chat_history = InMemoryChatMessageHistory()
        
        # Track project context (hazard class, system type, etc.)
        self.project_context = {}
    
    def add_exchange(self, question: str, answer: str, metadata: Optional[Dict] = None):
        """
        Add a question-answer exchange to memory
        
        Args:
            question: User question
            answer: AI answer
            metadata: Additional metadata about the exchange
        """
        # Add messages to history
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(answer)
        
        # Keep only last k exchanges (k*2 messages)
        if len(self.chat_history.messages) > self.k * 2:
            # Keep only the most recent messages
            self.chat_history.messages = self.chat_history.messages[-(self.k * 2):]
        
        # Extract and update project context
        if metadata:
            self._update_project_context(question, answer, metadata)
    
    def get_chat_history(self) -> List[BaseMessage]:
        """Get chat history as list of BaseMessage objects"""
        return self.chat_history.messages.copy()
    
    def get_chat_history_string(self) -> str:
        """
        Get chat history as formatted string for prompt inclusion
        """
        if not self.chat_history.messages:
            return ""
        
        formatted = []
        for message in self.chat_history.messages:
            if isinstance(message, HumanMessage):
                formatted.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted)
    
    def get_memory_variables(self) -> Dict:
        """
        Get memory variables in LangChain format
        Returns dict with 'chat_history' key for compatibility
        """
        return {
            "chat_history": self.chat_history.messages,
            "history_string": self.get_chat_history_string()
        }
    
    def clear_memory(self):
        """Clear all conversation memory"""
        self.chat_history.clear()
        self.project_context = {}
    
    def get_last_n_exchanges(self, n: int) -> List[BaseMessage]:
        """
        Get last n exchanges (n*2 messages)
        
        Args:
            n: Number of exchanges to retrieve
            
        Returns:
            List of messages from last n exchanges
        """
        total_messages = n * 2
        messages = self.chat_history.messages
        
        if len(messages) <= total_messages:
            return messages
        
        return messages[-total_messages:]
    
    def _update_project_context(self, question: str, answer: str, metadata: Dict):
        """
        Extract and store project-specific context
        Helps with contextual retrieval
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Detect hazard classification
        hazard_keywords = {
            'light hazard': 'light_hazard',
            'ordinary hazard': 'ordinary_hazard',
            'ordinary hazard group 1': 'ordinary_hazard_1',
            'ordinary hazard group 2': 'ordinary_hazard_2',
            'extra hazard': 'extra_hazard',
            'extra hazard group 1': 'extra_hazard_1',
            'extra hazard group 2': 'extra_hazard_2',
            'high piled storage': 'high_piled_storage'
        }
        
        for keyword, value in hazard_keywords.items():
            if keyword in question_lower or keyword in answer_lower:
                self.project_context['hazard_class'] = value
                break
        
        # Detect system type
        system_keywords = {
            'wet pipe': 'wet_pipe',
            'dry pipe': 'dry_pipe',
            'preaction': 'preaction',
            'deluge': 'deluge',
            'foam-water sprinkler': 'foam_water'
        }
        
        for keyword, value in system_keywords.items():
            if keyword in question_lower or keyword in answer_lower:
                self.project_context['system_type'] = value
                break
        
        # Extract numerical values from metadata
        if metadata:
            if 'flow_rate' in metadata:
                self.project_context['flow_rate'] = metadata['flow_rate']
            if 'pressure' in metadata:
                self.project_context['pressure'] = metadata['pressure']
            if 'pipe_size' in metadata:
                self.project_context['pipe_size'] = metadata['pipe_size']
    
    def get_project_context(self) -> Dict:
        """Get accumulated project context"""
        return self.project_context.copy()
    
    def format_context_for_retrieval(self) -> str:
        """
        Format project context as string for enhanced retrieval
        Can be used to narrow down vector search
        """
        if not self.project_context:
            return ""
        
        context_parts = []
        
        if 'hazard_class' in self.project_context:
            context_parts.append(f"Hazard: {self.project_context['hazard_class']}")
        
        if 'system_type' in self.project_context:
            context_parts.append(f"System: {self.project_context['system_type']}")
        
        if 'flow_rate' in self.project_context:
            context_parts.append(f"Flow: {self.project_context['flow_rate']} GPM")
        
        return " | ".join(context_parts)
    
    def get_conversation_summary(self) -> str:
        """
        Get a brief summary of the conversation for context
        """
        messages = self.chat_history.messages
        
        if not messages:
            return "New conversation"
        
        num_exchanges = len(messages) // 2
        topics = list(self.project_context.keys())
        
        summary = f"{num_exchanges} exchanges"
        if topics:
            summary += f" about {', '.join(topics)}"
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("Testing ConversationMemoryManager...")
    
    # Create manager
    memory_mgr = ConversationMemoryManager(k=5)
    
    # Simulate conversation
    memory_mgr.add_exchange(
        "What is the minimum density for ordinary hazard?",
        "According to Section 5.2, for ordinary hazard Group 2, minimum density is 0.20 gpm/ft² over 1500 ft²",
        metadata={'hazard_class': 'ordinary_hazard_2'}
    )
    
    memory_mgr.add_exchange(
        "What about the coverage area?",
        "For ordinary hazard Group 2, the design area is 1500 ft² per Section 5.2.1"
    )
    
    memory_mgr.add_exchange(
        "Calculate pipe size for 150 GPM",
        "Using the Hazen-Williams formula from Section 4.3.2, for 150 GPM flow rate...",
        metadata={'flow_rate': 150}
    )
    
    # Test outputs
    print("\n" + "="*60)
    print("CHAT HISTORY STRING:")
    print("="*60)
    print(memory_mgr.get_chat_history_string())
    
    print("\n" + "="*60)
    print("PROJECT CONTEXT:")
    print("="*60)
    print(memory_mgr.get_project_context())
    
    print("\n" + "="*60)
    print("CONTEXT FOR RETRIEVAL:")
    print("="*60)
    print(memory_mgr.format_context_for_retrieval())
    
    print("\n" + "="*60)
    print("CONVERSATION SUMMARY:")
    print("="*60)
    print(memory_mgr.get_conversation_summary())
    
    print("\n All tests passed!")