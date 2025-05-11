import logging
from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EpistemologyType(Enum):
    WESTERN_MAINSTREAM = "western_mainstream"
    DECOLONIAL = "decolonial"

@dataclass
class Response:
    text: str
    epistemology_type: EpistemologyType
    assumptions: List[str]
    power_dynamics: List[str]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "text": self.text,
            "epistemology_type": self.epistemology_type.value,
            "assumptions": self.assumptions,
            "power_dynamics": self.power_dynamics,
            "timestamp": self.timestamp
        }

class AIAgent:
    def __init__(self, epistemology_type: EpistemologyType):
        self.epistemology_type = epistemology_type
        self.conversation_history: List[Response] = []
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        
        # Knowledge bases for different epistemological frameworks
        self.knowledge_base = {
            EpistemologyType.WESTERN_MAINSTREAM: {
                "assumptions": [
                    "Universal truth exists and can be discovered through scientific method",
                    "Knowledge is objective and value-neutral",
                    "Linear progress and development are natural",
                    "Individual rationality is the primary mode of knowing",
                    "Quantitative methods are superior to qualitative ones"
                ],
                "response_guidelines": [
                    "Emphasize empirical evidence and scientific method",
                    "Focus on universal principles and objective truth",
                    "Highlight individual agency and rationality",
                    "Reference established institutions and protocols",
                    "Maintain value-neutral analysis"
                ]
            },
            EpistemologyType.DECOLONIAL: {
                "assumptions": [
                    "Knowledge is situated and contextual",
                    "Multiple ways of knowing are valid",
                    "Power shapes what counts as knowledge",
                    "Community and relational knowledge are essential",
                    "Historical and cultural context matters"
                ],
                "response_guidelines": [
                    "Emphasize contextual understanding",
                    "Highlight multiple valid perspectives",
                    "Address power dynamics and inequalities",
                    "Center community and relational knowledge",
                    "Consider historical and cultural context"
                ]
            }
        }
        logger.info(f"Initialized AIAgent with {epistemology_type.value} epistemology")

    def generate_response(self, input_text: str) -> Response:
        try:
            # Determine response type
            response_type = "knowledge" if any(word in input_text.lower() 
                for word in ["know", "learn", "understand", "truth"]) else "power"
            
            # Generate response using GPT-4
            response_text = self._generate_gpt4_response(input_text, response_type)
            
            # Identify relevant assumptions using GPT-4
            relevant_assumptions = self._identify_relevant_assumptions(input_text, response_text)
                
            # Analyze power dynamics
            power_dynamics = self._analyze_power_dynamics(input_text, response_text)
            
            response = Response(
                text=response_text,
                epistemology_type=self.epistemology_type,
                assumptions=relevant_assumptions,
                power_dynamics=power_dynamics
            )
            
            self.conversation_history.append(response)
            logger.info(f"Generated response for {self.epistemology_type.value} agent")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return Response(
                text="I need to reflect on this further.",
                epistemology_type=self.epistemology_type,
                assumptions=[self.knowledge_base[self.epistemology_type]["assumptions"][0]],
                power_dynamics=["Error in response generation"]
            )

    def _generate_gpt4_response(self, input_text: str, response_type: str) -> str:
        try:
            # Prepare system message with epistemological guidelines
            guidelines = self.knowledge_base[self.epistemology_type]["response_guidelines"]
            
            # Enhanced system message with epistemology-specific instructions
            if self.epistemology_type == EpistemologyType.WESTERN_MAINSTREAM:
                system_message = f"""You are an AI agent operating within the Western Mainstream epistemological framework.
                
                Core epistemological stance:
                {', '.join(guidelines)}
                
                Your role is to engage in dialogue while maintaining these key principles:
                1. Knowledge is discovered through systematic, empirical investigation
                2. Truth claims must be verifiable and falsifiable
                3. Individual reason and evidence are primary sources of knowledge
                4. Universal principles can be derived from particular observations
                5. Progress is measured through objective, quantifiable outcomes
                
                Response requirements:
                1. Be concise and direct (3-5 sentences maximum)
                2. Ground your response in empirical evidence or logical reasoning
                3. Maintain objectivity and value-neutrality
                4. Reference established scientific or academic frameworks
                5. Focus on universal applicability of your insights
                
                Remember: Your response should reflect the Western scientific tradition's emphasis on empirical verification and universal truth."""
            else:  # DECOLONIAL
                system_message = f"""You are an AI agent operating within the Decolonial epistemological framework.
                
                Core epistemological stance:
                {', '.join(guidelines)}
                
                Your role is to engage in dialogue while maintaining these key principles:
                1. Knowledge is situated within specific historical and cultural contexts
                2. Multiple ways of knowing are equally valid
                3. Power structures shape what counts as legitimate knowledge
                4. Community and relational understanding are essential
                5. Knowledge production must address historical injustices
                
                Response requirements:
                1. Be concise and direct (3-5 sentences maximum)
                2. Acknowledge the contextual nature of knowledge
                3. Consider whose perspectives are centered or marginalized
                4. Highlight alternative ways of knowing
                5. Connect insights to broader power structures
                
                Remember: Your response should reflect decolonial commitment to multiple knowledges and challenging dominant epistemologies."""
            
            # Prepare conversation history for context
            conversation_context = ""
            if self.conversation_history:
                conversation_context = "\nPrevious context:\n"
                for response in self.conversation_history[-2:]:  # Include last 2 exchanges
                    conversation_context += f"{response.text}\n"
            
            # Generate response using GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{conversation_context}\nInput: {input_text}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in GPT-4 response generation: {str(e)}", exc_info=True)
            raise

    def _identify_relevant_assumptions(self, input_text: str, response_text: str) -> List[str]:
        try:
            # Get all assumptions for this epistemology
            all_assumptions = self.knowledge_base[self.epistemology_type]["assumptions"]
            
            system_message = f"""You are analyzing a response from a {self.epistemology_type.value} epistemological perspective.
            
            Available assumptions for this framework:
            {chr(10).join(f'- {assumption}' for assumption in all_assumptions)}
            
            Your task is to identify 2-3 most relevant assumptions that underlie or inform this response. Consider:
            - Which assumptions are most directly relevant to the input and response
            - How the response implicitly or explicitly relies on these assumptions
            - The epistemological framework's core principles
            
            IMPORTANT: You must select assumptions EXACTLY as they appear in the list above.
            Do not modify, rephrase, or create new assumptions.
            
            Return ONLY a list of 2-3 assumptions from the provided list, one per line, without any additional text or explanation."""
            
            analysis = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Input: {input_text}\nResponse: {response_text}"}
                ],
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            # Get the raw response
            raw_response = analysis.choices[0].message.content.strip()
            
            # Parse the response into a list of assumptions
            selected_assumptions = [
                line.strip() 
                for line in raw_response.split('\n')
                if line.strip()
            ]
            
            # Validate that the selected assumptions are from our list
            valid_assumptions = []
            for assumption in selected_assumptions:
                # Try exact match first
                if assumption in all_assumptions:
                    valid_assumptions.append(assumption)
                else:
                    # Try fuzzy matching for close matches
                    for original in all_assumptions:
                        if self._is_similar_assumption(assumption, original):
                            valid_assumptions.append(original)
                            break
            
            # If no valid assumptions were found, use semantic matching
            if not valid_assumptions:
                semantic_matches = self._find_semantic_matches(input_text, response_text, all_assumptions)
                if semantic_matches:
                    valid_assumptions = semantic_matches[:3]
                else:
                    # Fallback to keyword matching
                    fallback_assumptions = [
                        a for a in all_assumptions 
                        if any(word in a.lower() for word in input_text.lower().split())
                    ]
                    if fallback_assumptions:
                        valid_assumptions = [fallback_assumptions[0]]
                    else:
                        valid_assumptions = [all_assumptions[0]]
            
            return valid_assumptions[:3]  # Return at most 3 assumptions
            
        except Exception as e:
            logger.error(f"Error identifying assumptions: {str(e)}", exc_info=True)
            # Fallback to simple keyword matching
            fallback_assumptions = [
                a for a in all_assumptions 
                if any(word in a.lower() for word in input_text.lower().split())
            ]
            return fallback_assumptions[:1] if fallback_assumptions else [all_assumptions[0]]

    def _is_similar_assumption(self, assumption1: str, assumption2: str) -> bool:
        """Check if two assumptions are similar enough to be considered the same."""
        # Convert to lowercase and remove punctuation
        a1 = ''.join(c.lower() for c in assumption1 if c.isalnum() or c.isspace())
        a2 = ''.join(c.lower() for c in assumption2 if c.isalnum() or c.isspace())
        
        # Check if one is contained within the other
        if a1 in a2 or a2 in a1:
            return True
            
        # Check word overlap
        words1 = set(a1.split())
        words2 = set(a2.split())
        overlap = words1.intersection(words2)
        
        # If more than 70% of words overlap, consider them similar
        return len(overlap) / max(len(words1), len(words2)) > 0.7
    
    def _find_semantic_matches(self, input_text: str, response_text: str, all_assumptions: List[str]) -> List[str]:
        """Use GPT-4 to find semantic matches between the response and assumptions."""
        system_message = f"""You are analyzing a response to find which of these assumptions best match its meaning:
        
        {chr(10).join(f'- {assumption}' for assumption in all_assumptions)}
        
        Return ONLY the numbers of the 2-3 most relevant assumptions (e.g., "1,3,4").
        Do not include any other text."""
        
        try:
            analysis = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Input: {input_text}\nResponse: {response_text}"}
                ],
                temperature=0.3
            )
            
            # Parse the numbers and get corresponding assumptions
            numbers = [
                int(n.strip()) 
                for n in analysis.choices[0].message.content.strip().split(',')
                if n.strip().isdigit()
            ]
            
            # Convert 1-based indices to 0-based and get assumptions
            return [all_assumptions[i-1] for i in numbers if 0 < i <= len(all_assumptions)]
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {str(e)}", exc_info=True)
            return []
    
    def _analyze_power_dynamics(self, input_text: str, response_text: str) -> List[str]:
        try:
            # Use GPT-4 to analyze power dynamics with epistemology-specific focus
            if self.epistemology_type == EpistemologyType.WESTERN_MAINSTREAM:
                system_message = f"""You are analyzing a response from a Western Mainstream epistemological perspective.
                
                Your task is to identify 2-3 key power dynamics present in the response. Focus on:
                - How scientific authority and expertise are positioned
                - The relationship between individual reason and institutional knowledge
                - Assumptions about universal applicability of knowledge
                - The role of empirical verification in establishing authority
                - How objectivity claims may mask particular perspectives
                
                Return ONLY a list of 2-3 concise power dynamics, one per line.
                Do not include numbers, bullet points, or any other formatting.
                Each dynamic should be a single clear statement without any prefixes."""
            else:  # DECOLONIAL
                system_message = f"""You are analyzing a response from a Decolonial epistemological perspective.
                
                Your task is to identify 2-3 key power dynamics present in the response. Focus on:
                - How colonial knowledge hierarchies are maintained or challenged
                - The centering or marginalization of different ways of knowing
                - The relationship between knowledge and power structures
                - How historical and cultural context shapes knowledge production
                - The role of community and relational understanding
                
                Return ONLY a list of 2-3 concise power dynamics, one per line.
                Do not include numbers, bullet points, or any other formatting.
                Each dynamic should be a single clear statement without any prefixes."""
            
            analysis = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Input: {input_text}\nResponse: {response_text}"}
                ],
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            # Parse the response into a list of dynamics
            dynamics = [
                line.strip().lstrip('1234567890.- ')  # Remove any numbers, dots, or dashes at start
                for line in analysis.choices[0].message.content.strip().split('\n')
                if line.strip()
            ]
            
            # Ensure we have at least one dynamic
            if not dynamics:
                dynamics = ["Power dynamics require deeper analysis"]
                
            return dynamics[:3]  # Return at most 3 dynamics
            
        except Exception as e:
            logger.error(f"Error analyzing power dynamics: {str(e)}", exc_info=True)
            return ["Power dynamics analysis unavailable"]

def save_conversation(topic: str, western_responses: List[Response], decolonial_responses: List[Response]) -> str:
    """Save the conversation to a JSON file."""
    try:
        conversation_data = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "exchanges": []
        }
        
        # Pair up the responses
        for w_resp, d_resp in zip(western_responses, decolonial_responses):
            exchange = {
                "western_response": w_resp.to_dict(),
                "decolonial_response": d_resp.to_dict()
            }
            conversation_data["exchanges"].append(exchange)
        
        # Create conversations directory if it doesn't exist
        os.makedirs("conversations", exist_ok=True)
        
        # Create filename with timestamp
        filename = f"conversations/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        logger.info(f"Saved conversation to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}", exc_info=True)
        raise

def simulate_conversation(topic: str, turns: int = 5):
    try:
        western_agent = AIAgent(EpistemologyType.WESTERN_MAINSTREAM)
        decolonial_agent = AIAgent(EpistemologyType.DECOLONIAL)
        
        logger.info(f"Starting conversation about: {topic}")
        print(f"\nStarting conversation about: {topic}\n")
        print("-" * 80)
        
        western_responses = []
        decolonial_responses = []
        
        current_input = topic
        for i in range(turns):
            # Western agent responds
            western_response = western_agent.generate_response(current_input)
            western_responses.append(western_response)
            print(f"\nWestern Mainstream Agent:")
            print(f"Response: {western_response.text}")
            print("\nAssumptions:")
            for assumption in western_response.assumptions:
                print(f"- {assumption}")
            print("\nPower Dynamics:")
            for dynamic in western_response.power_dynamics:
                print(f"- {dynamic}")
            
            # Decolonial agent responds
            decolonial_response = decolonial_agent.generate_response(western_response.text)
            decolonial_responses.append(decolonial_response)
            print(f"\nDecolonial Agent:")
            print(f"Response: {decolonial_response.text}")
            print("\nAssumptions:")
            for assumption in decolonial_response.assumptions:
                print(f"- {assumption}")
            print("\nPower Dynamics:")
            for dynamic in decolonial_response.power_dynamics:
                print(f"- {dynamic}")
            
            print("\n" + "-" * 80)
            current_input = decolonial_response.text
        
        # Save the conversation
        filename = save_conversation(topic, western_responses, decolonial_responses)
        print(f"\nConversation saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Error in conversation simulation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Example conversation topics
    topics = [
        # Knowledge and Truth
        "How do we determine what counts as valid knowledge in different cultural contexts?",
        
        # Progress and Development
        "How do we define progress without imposing Western standards on other cultures?",
        
        # Leadership and Power
        "How do different cultural traditions understand and practice leadership?",
        
        # Science and Tradition
        "How can we integrate traditional knowledge systems with modern scientific methods?",
                
        # Technology and Society
        "How do we ensure technological development serves diverse cultural values?",   
    ]
    
    # Simulate a conversation for each topic
    for topic in topics:
        simulate_conversation(topic)
        print("\n" + "=" * 80 + "\n")
