import streamlit as st
import pandas as pd
from pypdf import PdfReader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
import os
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# If you need to use functions or classes from modules.py, use:
# from modules import *
# from modules import *

# --- API KEY SETUP ---
openai_key = st.secrets["api_keys"]["openai"]
os.environ["OPENAI_API_KEY"] = openai_key

# --- FILE PATHS ---
policy_manual_path = "./data/policies_and_procedure_manual_2022.pdf"
training_data_path = "./data//UETCL_Training_Data_new.csv"
roles_and_departments_path = "./data/UETCL_Roles_and_Departments.csv"

# --- DATA MODELS FOR CLASSIFICATION ---
class UserIntent(BaseModel):
    intent: str = Field(description="Classify the user's intent as either 'greeting' or 'substantive_question'.")
class UserIntent(BaseModel):
    primary_intent: str = Field(description="Primary intent: 'question', 'continue', 'challenge_response', 'help', 'general_chat'")
    confidence: float = Field(description="Confidence level 0-1")
    requires_module_context: bool = Field(description="Whether this requires current module context")
    topic_keywords: List[str] = Field(description="Key topics mentioned")

class ConversationContext(BaseModel):
    current_topic: str = Field(default="general")
    module_active: bool = Field(default=False)
    last_user_intent: str = Field(default="")
    conversation_history: List[str] = Field(default_factory=list)

# --- ROLE-BASED ENHANCEMENTS ---
class RiskLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    STANDARD = "standard"

class TechnicalLevel(Enum):
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BASIC = "basic"

@dataclass
class RoleProfile:
    role: str
    department: str
    risk_level: RiskLevel
    technical_level: TechnicalLevel
    mandatory_modules: List[str]
    recommended_modules: List[str]
    scenario_focus: List[str]
    description: str

# --- ROLE MAPPING (Based on your CSV data) ---
ROLE_PROFILES = {
    # IT Department - High Risk
    "IT Technician": RoleProfile(
        role="IT Technician",
        department="Information and Communication Technology",
        risk_level=RiskLevel.HIGH,
        technical_level=TechnicalLevel.ADVANCED,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 4", "Module 7", "Module 8", "Module 9"],
        recommended_modules=["Module 5", "Module 6", "Module 10"],
        scenario_focus=["network_security", "system_administration", "technical_controls"],
        description="You manage critical IT infrastructure and have elevated system access."
    ),
    
    "Manager IT": RoleProfile(
        role="Manager IT",
        department="Information and Communication Technology",
        risk_level=RiskLevel.HIGH,
        technical_level=TechnicalLevel.ADVANCED,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 4", "Module 7", "Module 8", "Module 9", "Module 10"],
        recommended_modules=["Module 5", "Module 6"],
        scenario_focus=["leadership", "incident_management", "policy_enforcement"],
        description="You lead the IT team and are responsible for organizational security policies."
    ),
    
    "IT Support Officer": RoleProfile(
        role="IT Support Officer",
        department="Information and Communication Technology", 
        risk_level=RiskLevel.HIGH,
        technical_level=TechnicalLevel.INTERMEDIATE,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 7", "Module 8", "Module 9"],
        recommended_modules=["Module 4", "Module 5", "Module 6", "Module 10"],
        scenario_focus=["user_support", "device_management", "basic_security"],
        description="You provide technical support and have access to user systems."
    ),
    
    # Finance Department - High Risk
    "Financial Accountant": RoleProfile(
        role="Financial Accountant",
        department="Finance",
        risk_level=RiskLevel.HIGH,
        technical_level=TechnicalLevel.INTERMEDIATE,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 4", "Module 5"],
        recommended_modules=["Module 6", "Module 7", "Module 8", "Module 10"],
        scenario_focus=["financial_data", "business_email_compromise", "regulatory_compliance"],
        description="You handle sensitive financial data and payment processing."
    ),
    
    # Operations - Medium Risk
    "Control Engineer": RoleProfile(
        role="Control Engineer",
        department="Operations and Maintenance",
        risk_level=RiskLevel.MEDIUM,
        technical_level=TechnicalLevel.INTERMEDIATE,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 6", "Module 7"],
        recommended_modules=["Module 4", "Module 5", "Module 8"],
        scenario_focus=["operational_systems", "field_security", "remote_operations"],
        description="You operate critical power systems and control infrastructure."
    ),
    
    # HR Department - Medium Risk
    "Human Resource Officer": RoleProfile(
        role="Human Resource Officer",
        department="Human Resource and Administration",
        risk_level=RiskLevel.MEDIUM,
        technical_level=TechnicalLevel.BASIC,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 4", "Module 10"],
        recommended_modules=["Module 5", "Module 6", "Module 8"],
        scenario_focus=["personal_data", "social_engineering", "hr_processes"],
        description="You handle employee personal information and confidential HR data."
    ),
    
    # Administration - Standard Risk
    "Administration Officer": RoleProfile(
        role="Administration Officer", 
        department="Human Resource and Administration",
        risk_level=RiskLevel.STANDARD,
        technical_level=TechnicalLevel.BASIC,
        mandatory_modules=["Module 1", "Module 2", "Module 3", "Module 5", "Module 10"],
        recommended_modules=["Module 4", "Module 6"],
        scenario_focus=["office_security", "basic_awareness", "policy_compliance"],
        description="You handle general administrative tasks and office coordination."
    )
}

# --- ROLE-SPECIFIC SCENARIOS ---
ROLE_SPECIFIC_SCENARIOS = {
    "Module 1": {
        "IT Technician": {
            "scenario": "You receive an email claiming to be from Microsoft Security, requesting immediate verification of server credentials due to 'SQL injection attempts detected'. The email includes technical jargon and a link to verify. What technical indicators should you check first?",
            "focus": "Email headers, domain verification, technical authenticity",
            "hint": "Look for technical inconsistencies and verify through official Microsoft channels"
        },
        "Financial Accountant": {
            "scenario": "An email appears to be from your CEO requesting an urgent wire transfer of $50,000 to a 'confidential acquisition target'. The email mentions a tight deadline. What financial controls should you follow?",
            "focus": "Business email compromise, financial verification procedures",
            "hint": "Always verify financial requests through established dual-authorization channels"
        },
        "Administration Officer": {
            "scenario": "An email claims your employee benefits account will be suspended unless you click a link to 'verify your information within 24 hours'. The email looks official but urgent. What should you do?",
            "focus": "Basic phishing recognition, reporting procedures",
            "hint": "Legitimate systems rarely require urgent action via email links"
        }
    }
}

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UETCL AI Security Advisor", page_icon="🛡️", layout="wide")

# --- MODULE CONTENT DEFINITIONS ---
module_1_content = [
    {"type": "instruction", "content": "Welcome to **Module 1: Phishing & Social Engineering Awareness!** In the next 5 minutes, you'll learn how to spot and report these common threats to protect both yourself and UETCL."},
    {"type": "instruction", "content": "Phishing is a fraudulent attempt to obtain sensitive information by disguising as a trustworthy entity. Key signs include a sense of urgency, generic greetings, mismatched URLs, and poor grammar."},
    {"type": "qa_prompt", "content": "What questions do you have about phishing? **When ready to continue, type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "An email arrives with the subject **'URGENT: Your Email Account Storage is Full'** from **'UETCL IT Support <IT.Helpdesk@uetcl-logins.com>'**. It demands you click a link within one hour. What is the correct action?",
        "correct_answer_keyword": "report"
    }},
    {"type": "final", "content": "Congratulations, you have completed the Phishing & Social Engineering Awareness module!"}
]

module_2_content = [
    {"type": "instruction", "content": "Welcome to **Module 2: Mastering Password & Access Control**! This module covers the most important rules for keeping your account secure."},
    {"type": "instruction", "content": "According to UETCL policy, your password **must** be at least **12 characters long** and contain a mix of **three** of the following four types: uppercase letters, lowercase letters, numbers, and special characters."},
    {"type": "instruction", "content": "Furthermore, you must change your password every **42 days**, and you cannot reuse any of your last five passwords. Never share your password with anyone, including IT staff."},
    {"type": "qa_prompt", "content": "What questions do you have about these rules? **When you are ready to continue, type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "Let's test your knowledge. A colleague tells you they use the password **'UetclRocks!23'**. Does this password comply with UETCL policy?",
        "correct_answer_keyword": "yes"
    }},
    {"type": "final", "content": "Excellent work! You have completed the Password & Access Control module."}
]

module_3_content = [
    {"type": "instruction", "content": "Welcome to **Module 3: Incident Reporting & Response**! A security incident is any event that violates security policy, like a virus, lost device, or unauthorized access. Your role in reporting these incidents quickly is crucial."},
    {"type": "instruction", "content": "According to UETCL policy, all suspected security incidents **must be reported immediately** to the **ICT Helpdesk**. Do not attempt to investigate or fix the issue yourself, as this can sometimes cause more damage."},
    {"type": "qa_prompt", "content": "Do you have any questions about what to report or how? **When you are ready to continue, just type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "You find a USB flash drive labeled 'Q3 Finances' in the parking lot. What is the correct action to take according to the incident response policy?",
        "correct_answer_keyword": "report"
    }},
    {"type": "final", "content": "You have successfully completed the Incident Reporting & Response module. Remember, fast reporting is key to security!"}
]

module_4_content = [
    {"type": "instruction", "content": "Welcome to **Module 4: Data Handling & Information Classification**! This module explains how to handle UETCL data based on its sensitivity."},
    {"type": "instruction", "content": "UETCL classifies data into three levels: **Confidential**, **Restricted**, and **Public**. 'Confidential' is the most sensitive data, while 'Public' data is approved for release to everyone. You must handle data according to its classification level."},
    {"type": "qa_prompt", "content": "Ask me any questions you have about the data classification levels. **When you are ready to continue, type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "A colleague from another department asks you to email them a customer list, which is classified as 'Restricted'. What is the first thing you should verify before sending it?",
        "correct_answer_keyword": "authorized"
    }},
    {"type": "final", "content": "Great work! You've completed the Data Handling module. Proper classification protects us all."}
]

module_5_content = [
    {"type": "instruction", "content": "Welcome to **Module 5: Safe Internet & Email Use**! This module covers the acceptable use of UETCL's digital resources."},
    {"type": "instruction", "content": "UETCL's internet and email are provided for official company business. Incidental personal use is permitted but should not interfere with your work. Accessing or distributing offensive or illegal material is strictly prohibited."},
    {"type": "qa_prompt", "content": "What questions do you have about the acceptable use policy? **When ready, type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "You used the office internet to download a large movie file for personal viewing after work. Which part of the policy might this action violate?",
        "correct_answer_keyword": "personal use"
    }},
    {"type": "final", "content": "You have completed the Safe Internet & Email Use module. Thank you for using our resources responsibly."}
]

module_6_content = [
    {"type": "instruction", "content": "Welcome to **Module 6: Physical & Environmental Security**! Protecting our physical assets is as important as our digital ones."},
    {"type": "instruction", "content": "Key policies include wearing your ID badge at all times, escorting all visitors, and maintaining a 'clean desk' policy, which means securing sensitive documents when you are away from your desk."},
    {"type": "qa_prompt", "content": "Feel free to ask any questions about physical security. **To continue, type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "You are leaving your desk for a 30-minute meeting. There is a printed document marked 'Confidential' on your desk. What should you do with it?",
        "correct_answer_keyword": "lock"
    }},
    {"type": "final", "content": "Module complete! A secure building starts with all of us."}
]

module_7_content = [
    {"type": "instruction", "content": "Welcome to **Module 7: Secure Remote Access**! This module covers how to work safely from outside the UETCL office."},
    {"type": "instruction", "content": "The only approved method for accessing the internal UETCL network from an external location is through the company-provided **Virtual Private Network (VPN)**. Connecting directly from public Wi-Fi is not secure and is prohibited."},
    {"type": "qa_prompt", "content": "Ask away with any remote access questions. **When you're ready, type 'continue'.**"},
    {"type": "challenge", "content": {
        "prompt": "You are working from a coffee shop using their password-protected Wi-Fi. To access your files on the UETCL server, is connecting to the Wi-Fi enough?",
        "correct_answer_keyword": "no"
    }},
    {"type": "final", "content": "You've completed the Secure Remote Access module. Stay safe out there!"}
]

module_8_content = [
    {"type": "instruction", "content": "Welcome to **Module 8: Mobile & Personal Device Security**! Let's cover security for devices on the go."},
    {"type": "instruction", "content": "If a UETCL-owned mobile device is lost or stolen, you must report it to the ICT Helpdesk as a security incident **immediately**. For personally-owned devices, connecting to the internal network requires authorization and compliance with security standards."},
    {"type": "qa_prompt", "content": "I'm here for any questions on mobile security. **Type 'continue' to proceed.**"},
    {"type": "challenge", "content": {
        "prompt": "You realize you left your company-issued tablet in a taxi. You think it will probably be turned in, so you decide to wait a day before reporting it. Is this the correct procedure?",
        "correct_answer_keyword": "no"
    }},
    {"type": "final", "content": "You have completed the Mobile & Personal Device Security module!"}
]

module_9_content = [
    {"type": "instruction", "content": "Welcome to **Module 9: Software Management & Licensing**! This module is about the safe and legal use of software."},
    {"type": "instruction", "content": "You are prohibited from installing any unlicensed, unauthorized, or personal software on UETCL computers. All software must be approved and installed by the ICT department to avoid security risks and legal issues."},
    {"type": "qa_prompt", "content": "Any questions about software policy? **Type 'continue' to proceed.**"},
    {"type": "challenge", "content": {
        "prompt": "You find a free, open-source note-taking app that you love. Can you install it on your work laptop yourself?",
        "correct_answer_keyword": "no"
    }},
    {"type": "final", "content": "Module complete. Thank you for helping UETCL maintain a secure and compliant software environment."}
]

module_10_content = [
    {"type": "instruction", "content": "Welcome to the final module, **Module 10: Social Media & Public Representation**! This covers how we represent UETCL online."},
    {"type": "instruction", "content": "When using social media, you must not disclose any confidential UETCL information. Always maintain a professional tone when discussing work-related matters, and make it clear when you are speaking in a personal capacity versus as a representative of the company."},
    {"type": "qa_prompt", "content": "Ask me anything about the social media policy. **Type 'continue' to finish.**"},
    {"type": "challenge", "content": {
        "prompt": "You have a disagreement with a UETCL business partner and post a frustrated comment about them on your private LinkedIn page. Could this violate company policy?",
        "correct_answer_keyword": "yes"
    }},
    {"type": "final", "content": "Congratulations, you have completed the entire cybersecurity training curriculum! Your dedication to security is appreciated."}
]

ALL_MODULES = {
    "Module 1: Phishing & Social Engineering": module_1_content,
    "Module 2: Password & Access Control": module_2_content,
    "Module 3: Incident Reporting & Response": module_3_content,
    "Module 4: Data Handling & Classification": module_4_content,
    "Module 5: Safe Internet & Email Use": module_5_content,
    "Module 6: Physical & Environmental Security": module_6_content,
    "Module 7: Secure Remote Access": module_7_content,
    "Module 8: Mobile & Personal Device Security": module_8_content,
    "Module 9: Software Management & Licensing": module_9_content,
    "Module 10: Social Media & Public Representation": module_10_content,
}

# --- ROLE-BASED HELPER FUNCTIONS ---
def get_available_roles() -> List[str]:
    """Get list of available roles for dropdown"""
    return sorted(list(ROLE_PROFILES.keys())) + ["Other (Please specify)"]

def get_user_profile(role: str, custom_role: str = None) -> Optional[RoleProfile]:
    """Get role profile or create default for custom roles"""
    if role == "Other (Please specify)" and custom_role:
        return create_custom_profile(custom_role)
    return ROLE_PROFILES.get(role)

def create_custom_profile(custom_role: str) -> RoleProfile:
    """Create profile for custom role based on keywords"""
    role_lower = custom_role.lower()
    
    if any(keyword in role_lower for keyword in ['it', 'technical', 'engineer', 'system']):
        risk_level = RiskLevel.HIGH
        technical_level = TechnicalLevel.ADVANCED
        mandatory_modules = ["Module 1", "Module 2", "Module 3", "Module 4", "Module 7", "Module 8", "Module 9"]
    elif any(keyword in role_lower for keyword in ['finance', 'accounting', 'commercial']):
        risk_level = RiskLevel.HIGH
        technical_level = TechnicalLevel.INTERMEDIATE
        mandatory_modules = ["Module 1", "Module 2", "Module 3", "Module 4", "Module 5"]
    elif any(keyword in role_lower for keyword in ['manager', 'director', 'head', 'senior']):
        risk_level = RiskLevel.MEDIUM
        technical_level = TechnicalLevel.INTERMEDIATE
        mandatory_modules = ["Module 1", "Module 2", "Module 3", "Module 4", "Module 10"]
    else:
        risk_level = RiskLevel.STANDARD
        technical_level = TechnicalLevel.BASIC
        mandatory_modules = ["Module 1", "Module 2", "Module 3", "Module 5"]
    
    return RoleProfile(
        role=custom_role,
        department="Custom/Other",
        risk_level=risk_level,
        technical_level=technical_level,
        mandatory_modules=mandatory_modules,
        recommended_modules=["Module 6", "Module 10"],
        scenario_focus=["general_awareness"],
        description=f"Custom role: {custom_role}"
    )
# 2. Intelligent Response Dispatcher
def classify_user_intent(user_input: str, context: dict) -> UserIntent:
    """Classify user intent using both keywords and context"""
    user_input_lower = user_input.lower().strip()
    
    # Direct continuation commands
    continue_keywords = ['continue', 'next', 'proceed', 'move on', 'go on']
    if any(keyword in user_input_lower for keyword in continue_keywords):
        return UserIntent(
            primary_intent="continue",
            confidence=0.9,
            requires_module_context=True,
            topic_keywords=[]
        )
    
    # Question indicators
    question_indicators = ['what', 'how', 'why', 'when', 'where', 'can you', 'could you', 'explain', '?']
    if any(indicator in user_input_lower for indicator in question_indicators):
        # Determine if module-specific or general
        module_active = context.get('module_active', False)
        current_module = context.get('selected_module', '')
        
        # Check if question relates to current module
        module_keywords = []
        if current_module:
            if 'phishing' in current_module.lower():
                module_keywords = ['phishing', 'email', 'social engineering', 'scam']
            elif 'password' in current_module.lower():
                module_keywords = ['password', 'authentication', 'login', 'access']
            # Add more module keyword mappings...
        
        requires_module_context = any(keyword in user_input_lower for keyword in module_keywords)
        
        return UserIntent(
            primary_intent="question",
            confidence=0.8,
            requires_module_context=requires_module_context,
            topic_keywords=module_keywords
        )
    
    # Challenge response (if we're in a challenge phase)
    if context.get('challenge_active', False):
        return UserIntent(
            primary_intent="challenge_response",
            confidence=0.7,
            requires_module_context=True,
            topic_keywords=[]
        )
    
    # Default to general chat
    return UserIntent(
        primary_intent="general_chat",
        confidence=0.5,
        requires_module_context=False,
        topic_keywords=[]
    )

# 3. Unified Response Handler
def intelligent_response_handler(user_input: str, context: dict, rag_retriever, llm, user_name: str, profile: RoleProfile = None):
    """Unified handler that can respond to any user input intelligently"""
    if profile is None:
        profile = create_custom_profile("General User")  # Or handle differently
    
    # Classify user intent
    intent = classify_user_intent(user_input, context)
    
    # Handle based on intent
    if intent.primary_intent == "continue":
        return handle_module_continuation(context)
    
    elif intent.primary_intent == "question":
        if context.get('module_active', False) and intent.requires_module_context:
            return handle_module_question(user_input, context, rag_retriever, llm, user_name, profile)
        else:
            return handle_general_question(user_input, rag_retriever, llm, user_name, profile)
    
    elif intent.primary_intent == "challenge_response":
        return handle_flexible_challenge_response(user_input, context, rag_retriever, llm, user_name, profile)
    
    else:  # general_chat or fallback
        return handle_conversational_response(user_input, context, rag_retriever, llm, user_name, profile)

# 4. Enhanced Module Question Handler
def handle_module_question(user_input: str, context: dict, rag_retriever, llm, user_name: str, profile: RoleProfile = None):
    """Handle questions specifically about the current module"""
    current_module = context.get('selected_module', '')
    module_step = context.get('module_step', 0)

    role_context = f" The user is a {profile.role} in {profile.department}." if profile else ""
    
    # Get relevant context from both current module and general knowledge
    policy_context = "\n\n".join([doc.page_content for doc in rag_retriever.get_relevant_documents(user_input)])
    
    # Build context-aware prompt
    role_context = f" The user is a {profile.role} in {profile.department}." if profile else ""
    module_context = f" They are currently in {current_module}, step {module_step + 1}."
    
    qa_template = f"""You are a UETCL cybersecurity tutor helping {user_name}.{role_context}{module_context}
    
They asked a question while going through their training module. Answer their question based on the policy context, 
keeping it relevant to their current module topic. Be conversational and encouraging.

After answering, let them know they can:
- Ask more questions about this topic
- Type 'continue' to proceed with the module
- Ask general cybersecurity questions anytime

Context: {{context}}
Current Module: {current_module}
User's Question: {{question}}
Answer:"""
    
    prompt_template = PromptTemplate(template=qa_template, input_variables=["context", "question"])
    qa_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = qa_chain.run(context=policy_context, question=user_input)

    
    return response

# 5. Flexible Challenge Response Handler
def handle_flexible_challenge_response(user_input: str, context: dict, rag_retriever, llm, user_name: str, profile: RoleProfile = None):
    """Handle challenge responses with flexibility and explanation"""
    module_content = context.get('current_module_content', [])
    module_step = context.get('module_step', 0)
    
    if module_step >= len(module_content):
        return "It looks like this module is complete. You can ask questions or select a new module."
    
    current_step = module_content[module_step]
    
    if current_step["type"] != "challenge":
        return "There's no active challenge right now. You can ask questions or type 'continue' to proceed."
    
    # Get the challenge details
    challenge_content = current_step["content"]
    correct_keyword = challenge_content["correct_answer_keyword"].lower()
    user_response = user_input.lower()
    
    # Check for direct keyword match first
    if correct_keyword in user_response:
        feedback = "✅ Excellent! You got it right."
        if profile and "focus" in challenge_content:
            feedback += f" As a {profile.role}, understanding {challenge_content['focus']} is particularly important for your role."
    else:
        # Use LLM to evaluate if the response shows understanding even without exact keywords
        evaluation_prompt = f"""
        Challenge: {challenge_content['prompt']}
        Correct concept: {correct_keyword}
        User response: {user_input}
        
        Does the user's response demonstrate understanding of the correct concept, even if they didn't use the exact keyword?
        Respond with either "CORRECT_UNDERSTANDING" or "NEEDS_CLARIFICATION" followed by a brief explanation.
        """
        
        policy_context = "\n\n".join([doc.page_content for doc in rag_retriever.get_relevant_documents(evaluation_prompt)])
        
        eval_template = PromptTemplate(
            template="Context: {context}\n\nEvaluation request: {prompt}\n\nEvaluation:",
            input_variables=["context", "prompt"]
        )
        eval_chain = LLMChain(llm=llm, prompt=eval_template)
        evaluation = eval_chain.run(context=policy_context, prompt=evaluation_prompt)
        
        if "CORRECT_UNDERSTANDING" in evaluation:
            feedback = "✅ Great! You demonstrate good understanding of the concept."
        else:
            feedback = f"❌ Not quite right. Let me explain: The key concept is '{correct_keyword}'."
            if profile and "hint" in challenge_content:
                feedback += f"\n\n💡 **Hint for {profile.role}s:** {challenge_content['hint']}"
    
    feedback += "\n\nYou can:\n- Ask follow-up questions about this challenge\n- Type 'continue' to finish the module\n- Ask any other cybersecurity questions"
    
    return feedback

# 6. Enhanced General Question Handler
def handle_general_question(user_input: str, rag_retriever, llm, user_name: str, profile: RoleProfile = None):
    """Handle general cybersecurity questions with role context"""
    policy_context = "\n\n".join([doc.page_content for doc in rag_retriever.get_relevant_documents(user_input)])
    
    role_context = f" The user is a {profile.role} in {profile.department}. Tailor your response to their role and responsibilities." if profile else ""
    
    qa_template = f"""You are a UETCL AI Cybersecurity Advisor helping {user_name}.{role_context}
    
Answer their question based on UETCL policies and cybersecurity best practices. Be conversational, helpful, and specific to their role when possible.

Context: {{context}}
Question: {{question}}
Answer:"""
    
    prompt_template = PromptTemplate(template=qa_template, input_variables=["context", "question"])
    qa_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = qa_chain.run(context=policy_context, question=user_input)
    
    return response

# 7. Conversational Response Handler
def handle_conversational_response(user_input: str, context: dict, rag_retriever, llm, user_name: str, profile: RoleProfile = None):
    """Handle general conversational inputs and provide helpful guidance"""
    
    # Check if user seems lost or needs help
    help_keywords = ['help', 'stuck', 'confused', 'what should i do', 'what now']
    if any(keyword in user_input.lower() for keyword in help_keywords):
        help_response = f"I'm here to help, {user_name}! "
        
        if context.get('module_active', False):
            current_module = context.get('selected_module', '')
            help_response += f"You're currently in {current_module}. You can:\n"
            help_response += "- Ask questions about the current topic\n"
            help_response += "- Type 'continue' to proceed with the module\n"
            help_response += "- Ask general cybersecurity questions\n"
            help_response += "- Use the 'Back to All Modules' button to choose a different module"
        else:
            help_response += "You can:\n"
            help_response += "- Select a training module from the options above\n"
            help_response += "- Ask any cybersecurity questions\n"
            help_response += "- Ask about UETCL security policies"
            
            if profile:
                help_response += f"\n\nAs a {profile.role}, I recommend focusing on your mandatory modules first."
        
        return help_response
    
    # For other conversational inputs, try to provide a helpful response
    return handle_general_question(user_input, rag_retriever, llm, user_name, profile)

# 8. Updated Module Continuation Handler
def handle_module_continuation(context: dict):
    """Handle module progression with better state management"""
    module_content = context.get('current_module_content', [])
    current_step = context.get('module_step', 0)
    
    # Move to next step
    next_step = current_step + 1
    
    if next_step < len(module_content):
        next_content = module_content[next_step]
        context['module_step'] = next_step
        
        if next_content["type"] == "challenge":
            context['challenge_active'] = True
            return next_content["content"]["prompt"]
        elif next_content["type"] == "final":
            # Mark module as completed
            selected_module = context.get('selected_module', '')
            if 'completed_modules' in st.session_state:
                st.session_state.completed_modules.add(selected_module)
            
            completion_msg = next_content["content"]
            profile = context.get('user_profile')
            if profile:
                completion_msg += f"\n\n**Great work, {profile.role}!** This training is specifically relevant to your role in {profile.department}."
            
            return completion_msg
        else:
            context['challenge_active'] = False
            return next_content.get("content", "Moving to the next step.")
    else:
        # Module completed
        context['module_active'] = False
        context['selected_module'] = None
        return "You have completed this module! You can now select another module or ask general questions."

# 9. Replace the main chat logic in your Streamlit app with this:
def handle_user_input(user_input: str):
    """Main handler for all user inputs - replace your existing chat logic with this"""
    profile = st.session_state.user_profile or create_custom_profile("General User")
    # Prepare context
    context = {
        'module_active': st.session_state.selected_module is not None,
        'selected_module': st.session_state.selected_module,
        'module_step': st.session_state.module_step,
        'current_module_content': st.session_state.get('current_module_content', []),
        'challenge_active': False,  # Set based on current step
        'user_profile': st.session_state.user_profile
    }
    
    # Check if we're in a challenge
    if context['module_active'] and context['module_step'] < len(context['current_module_content']):
        current_step = context['current_module_content'][context['module_step']]
        context['challenge_active'] = current_step.get("type") == "challenge"
    
    # Get intelligent response
    response = intelligent_response_handler(
        user_input, 
        context, 
        rag_retriever, 
        llm, 
        st.session_state.user_name, 
        st.session_state.user_profile
    )
    
    # Update session state based on context changes
    if 'module_step' in context:
        st.session_state.module_step = context['module_step']
    if 'selected_module' in context:
        st.session_state.selected_module = context['selected_module']
    if 'module_active' in context and not context['module_active']:
        st.session_state.selected_module = None
    
    return response

def customize_module_content(base_content: List, module_id: str, profile: RoleProfile = None) -> List:
    """Customize module content based on user's role"""
    customized = []
    
    for item in base_content:
        if item["type"] == "instruction":
            # Add role-specific context to instructions
            enhanced_instruction = add_role_context(item["content"], profile)
            customized.append({
                "type": "instruction",
                "content": enhanced_instruction
            })
            
        elif item["type"] == "challenge":
            # Use role-specific scenario if available
            if module_id in ROLE_SPECIFIC_SCENARIOS and profile.role in ROLE_SPECIFIC_SCENARIOS[module_id]:
                role_scenario = ROLE_SPECIFIC_SCENARIOS[module_id][profile.role]
                customized.append({
                    "type": "challenge",
                    "content": {
                        "prompt": role_scenario["scenario"],
                        "correct_answer_keyword": item["content"]["correct_answer_keyword"],
                        "focus": role_scenario["focus"],
                        "hint": role_scenario["hint"]
                    }
                })
            else:
                customized.append(item)
        else:
            customized.append(item)
    
    return customized

def add_role_context(instruction: str, profile: RoleProfile = None) -> str:
    """Add role-specific context to instruction text"""
    if not profile:
        return instruction
    
    role_intro = f"\n\n**👤 For {profile.role}s in {profile.department}:**\n"
    role_intro = f"\n\n**👤 For {profile.role}s in {profile.department}:**\n"
    role_intro += f"*{profile.description}*\n\n"
    
    if profile.technical_level == TechnicalLevel.ADVANCED:
        technical_note = "This module includes advanced technical concepts relevant to your technical responsibilities."
    elif profile.technical_level == TechnicalLevel.INTERMEDIATE:
        technical_note = "This module focuses on practical security measures for your daily work."
    else:
        technical_note = "This module covers essential security basics for your role."
    
    risk_context = ""
    if profile.risk_level == RiskLevel.HIGH:
        risk_context = "\n\n⚠️ **High Risk Role**: Your position involves access to sensitive systems or data."
    elif profile.risk_level == RiskLevel.MEDIUM:
        risk_context = "\n\n⚡ **Medium Risk Role**: Your role involves some sensitive information access."
    
    return instruction + role_intro + technical_note + risk_context

def display_training_dashboard_with_history(profile: RoleProfile = None):
    """Displays a dynamic dashboard with a real logo in the header."""

    # --- Header with Logo (New Design) ---
    # Create columns for the logo and the title
    col1, col2 = st.columns([1, 4], vertical_alignment="center")

    with col1:
        # Display your logo image. Adjust width as needed.
        # Ensure 'uetcl_logo.png' is in the same folder as your script.
        try:
            st.image("./static/uetcl_logo.jpg", width=60)
        except Exception:
            st.error("Logo not found!")

    with col2:
        # Display the title next to the logo
        st.title("Dashboard")


    # --- User Profile Section (No changes needed here) ---
    st.subheader("Your Profile")
    if profile:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Name**")
            with c2:
                st.write(st.session_state.user_name)
            st.divider()
            with c1:
                st.markdown("**Role**")
            with c2:
                st.write(profile.role)
            st.divider()
            with c1:
                st.markdown("**Department**")
            with c2:
                st.write(profile.department)
            st.divider()
    else:
         st.info(f"👤 User: **{st.session_state.user_name}**")

    st.markdown("---")

    # --- Progress Section (No changes needed here) ---
    st.subheader("📊 Your Progress")
    if profile and profile.mandatory_modules:
        completed_mandatory = {
            mod for mod in st.session_state.completed_modules 
            if mod.split(":")[0] in profile.mandatory_modules
        }
        total_mandatory = len(profile.mandatory_modules)
        progress = len(completed_mandatory) / total_mandatory if total_mandatory > 0 else 0
        st.metric(
            label="Mandatory Training",
            value=f"{len(completed_mandatory)} / {total_mandatory}",
            delta=f"{progress:.0%} Complete" if progress > 0 else "0% Complete"
        )
        st.progress(progress)
    else:
        st.write("No role profile loaded to track progress.")

    st.markdown("---")

    # --- History Section (No changes needed here) ---
    st.subheader("📚 Module History")
    if st.session_state.completed_modules:
        for module_name in sorted(list(st.session_state.completed_modules)):
            st.markdown(f"✅ &nbsp; {module_name}")
    else:
        st.markdown("_You haven't completed any modules yet._")
def get_personalized_modules(profile: RoleProfile = None) -> List[str]:
    """Get personalized module list based on role"""
    all_modules = profile.mandatory_modules + profile.recommended_modules
    module_list = []
    
    for module_id in all_modules:
        for full_module_name in ALL_MODULES.keys():
            if full_module_name.startswith(module_id):
                priority = "🔴 MANDATORY" if module_id in profile.mandatory_modules else "🟡 Recommended"
                module_list.append(f"{priority} - {full_module_name}")
                break
    
    return module_list

# --- APPLICATION CACHING ---
@st.cache_resource
def load_and_process_data():
    """Loads data and initializes the RAG pipeline. This runs only once."""
    reader = PdfReader(policy_manual_path)
    policy_text = "".join(page.extract_text() for page in reader.pages)
    training_df = pd.read_csv(training_data_path)
    training_qa_chunks = [f"Question: {row['Question']} Answer: {row['Answer']}" for _, row in training_df.iterrows()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    policy_chunks = text_splitter.split_text(policy_text)
    all_chunks = policy_chunks + training_qa_chunks
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=all_chunks, embedding=embedding_model)
    llm = OpenAI(temperature=0)
    rag_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return rag_retriever, llm

# --- ENHANCED RESPONSE HANDLER ---
def handle_role_based_qa_or_challenge_response(prompt, module_content, rag_retriever, llm, user_name, profile):
    """Enhanced version with role-based features"""
    current_step = module_content[st.session_state.module_step]
    
    # Check if user wants to continue to next step
    if prompt.lower().strip() == "continue":
        st.session_state.module_step += 1
        if st.session_state.module_step < len(module_content):
            next_step = module_content[st.session_state.module_step]
            if next_step["type"] == "challenge":
                return next_step["content"]["prompt"]
            elif next_step["type"] == "final":
                st.session_state.completed_modules.add(st.session_state.selected_module)
                # Add role-specific completion message
                completion_msg = next_step["content"]
                if profile:
                    completion_msg += f"\n\n**Great work, {profile.role}!** This training is specifically relevant to your role in {profile.department}."
                    
                    # Add role-specific next steps
                    remaining_mandatory = [m for m in profile.mandatory_modules if m not in [st.session_state.selected_module.split(":")[0]]]
                    if remaining_mandatory:
                        completion_msg += f"\n\n**Next Steps:** You still have {len(remaining_mandatory)} mandatory modules remaining for your role."
                
                return completion_msg
            else:
                return next_step.get("content", "Moving to the next step.")
        else:
            st.session_state.selected_module = None
            return "You have completed this module! You can now select another module or ask general questions."
    
    # Handle challenge responses with role-specific feedback
    elif current_step["type"] == "challenge":
        correct_keyword = current_step["content"]["correct_answer_keyword"].lower()
        user_response = prompt.lower()
        
        if correct_keyword in user_response:
            feedback = "✅ Correct! Well done."
            # Add role-specific praise
            if profile and "focus" in current_step["content"]:
                feedback += f" As a {profile.role}, understanding {current_step['content']['focus']} is crucial for your daily responsibilities."
            feedback += " **Type 'continue' to finish the module.**"
        else:
            feedback = f"❌ Not quite right. The key concept to remember is '{correct_keyword}'."
            # Add role-specific hint if available
            if profile and "hint" in current_step["content"]:
                feedback += f"\n\n💡 **Hint for {profile.role}s:** {current_step['content']['hint']}"
            feedback += " **Type 'continue' to finish the module.**"
        
        return feedback
    
    # Handle Q&A during qa_prompt phase with role context
    elif current_step["type"] == "qa_prompt":
        policy_context = "\n\n".join([doc.page_content for doc in rag_retriever.get_relevant_documents(prompt)])
        
        current_module = st.session_state.selected_module
        module_topic = current_module.split(":")[1].strip() if ":" in current_module else current_module
        
        # Enhanced template with role context
        role_context = ""
        if profile:
            role_context = f" The user is a {profile.role} in the {profile.department} department with {profile.technical_level.value} technical level."
        
        qa_template = f"""You are a UETCL cybersecurity tutor helping {user_name} with {module_topic}.{role_context}
Answer their question based on the UETCL policy context provided, keeping your response focused on this module's topic and relevant to their role.
Be conversational and helpful. End your response by reminding them they can ask more questions or type 'continue' when ready to proceed.

Context: {{context}}
User's Question: {{question}}
Answer:"""
        
        prompt_to_qa = PromptTemplate(template=qa_template, input_variables=["context", "question"])
        qa_chain = LLMChain(llm=llm, prompt=prompt_to_qa)
        response = qa_chain.run(context=policy_context, question=prompt)
        
        return response
    
    return "I'm not sure how to respond to that. Type 'continue' to proceed with the module."

# --- ORIGINAL HELPER FUNCTION (for backward compatibility) ---
def handle_qa_or_challenge_response(prompt, module_content, rag_retriever, llm, user_name):
    """Original function for backward compatibility"""
    # Use the enhanced function with no profile
    return handle_role_based_qa_or_challenge_response(prompt, module_content, rag_retriever, llm, user_name, None)

# --- MAIN APPLICATION ---
st.title("🛡️ UETCL AI Cybersecurity Tutor")

rag_retriever, llm = load_and_process_data()

# --- Initialize Session State ---
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = ""
if 'custom_role' not in st.session_state:
    st.session_state.custom_role = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_module" not in st.session_state:
    st.session_state.selected_module = None
if "module_step" not in st.session_state:
    st.session_state.module_step = 0
if "completed_modules" not in st.session_state:
    st.session_state.completed_modules = set()

# --- User Details Input Form ---
if not st.session_state.user_name:
    st.info("Welcome! Please enter your details to begin personalized cybersecurity training.")
    
    with st.form("user_details"):
        name = st.text_input("What is your first name?")
        
        # Role selection with dropdown
        available_roles = get_available_roles()
        selected_role = st.selectbox("What is your role at UETCL?", available_roles)
        
        # Custom role input if "Other" is selected
        custom_role = ""
        if selected_role == "Other (Please specify)":
            custom_role = st.text_input("Please specify your role:")
        
        submitted = st.form_submit_button("Start Personalized Training")
        
        if submitted:
            if name and selected_role:
                if selected_role == "Other (Please specify)" and not custom_role:
                    st.warning("Please specify your custom role.")
                else:
                    st.session_state.user_name = name
                    st.session_state.user_role = selected_role
                    st.session_state.custom_role = custom_role
                    
                    # Get user profile
                    profile = get_user_profile(selected_role, custom_role)
                    st.session_state.user_profile = profile
                    
                    # Create personalized welcome message
                    if profile:
                        welcome_msg = f"Hello {name}! As a {profile.role} in {profile.department}, you play a key role in our security. "
                        welcome_msg += f"Your training is customized for your {profile.risk_level.value} risk level role. "
                        welcome_msg += "Check the sidebar for your personalized training modules!"
                    else:
                        welcome_msg = f"Hello {name}! I'm here to help with your cybersecurity training. You can start by selecting a training module from the sidebar."
                    
                    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                    st.rerun()
            else:
                st.warning("Please provide both your name and role.")
else:
    # --- MAIN APP INTERFACE ---
    profile = st.session_state.user_profile

    # Function to inject custom CSS from the file
    def local_css(file_name):
        try:
            with open(file_name) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"CSS file '{file_name}' not found. Please create it for custom styling.")

    # Apply the custom CSS
    local_css("./static/style.css")

    # --- SIDEBAR: FOR DISPLAY ONLY ---
    with st.sidebar:
        if profile:
            display_training_dashboard_with_history(profile)
        else:
            st.info(f"👤 User: **{st.session_state.user_name}**")

    # --- TOP BAR BUTTON (NEW LOCATION) ---
    if st.session_state.selected_module:
        if st.button("⬅️ Back to All Modules", type="secondary"):
            st.session_state.selected_module = None
            st.session_state.messages = [{"role": "assistant", "content": f"Welcome back, {st.session_state.user_name}! Select a new module to begin or ask a general question."}]
            st.rerun()
        st.markdown("<hr style='margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)

    # --- CHAT HISTORY ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- MODULE SELECTION OR CHAT INTERFACE ---
    if st.session_state.selected_module is None:
        chip_container = st.container()
        with chip_container:
            current_profile = st.session_state.get('user_profile')
            
            # Initialize module lists with safe defaults
            mandatory_modules = []
            recommended_modules = []
            
            if current_profile and hasattr(current_profile, 'mandatory_modules'):
                mandatory_modules = [
                    name for mod_id in current_profile.mandatory_modules 
                    for name in ALL_MODULES 
                    if name.startswith(mod_id)
                ]
                
            if current_profile and hasattr(current_profile, 'recommended_modules'):
                recommended_modules = [
                    name for mod_id in current_profile.recommended_modules 
                    for name in ALL_MODULES 
                    if name.startswith(mod_id)
                ]
            
            if not mandatory_modules and not recommended_modules:
                mandatory_modules = list(ALL_MODULES.keys())

            def select_module(module_name):
                st.session_state.selected_module = module_name
                st.session_state.module_step = 0
                st.session_state.messages = []
                base_content = ALL_MODULES[module_name]
                current_profile = st.session_state.get('user_profile')
                if current_profile:
                    module_id = module_name.split(":")[0].strip()
                    customized_content = customize_module_content(base_content, module_id, current_profile)
                    st.session_state.current_module_content = customized_content
                    first_message = customized_content[0]["content"]
                else:
                    st.session_state.current_module_content = base_content
                    first_message = base_content[0]["content"]
                
                enhanced_message = first_message + "\n\n💬 **You can ask questions anytime during this module, or type 'continue' to proceed.**"
                st.session_state.messages.append({"role": "assistant", "content": enhanced_message})

            if mandatory_modules:
                st.markdown("##### 🔴 Mandatory Modules")
                cols = st.columns(3)
                for i, module_name in enumerate(mandatory_modules):
                    cols[i % 3].button(
                        module_name, 
                        key=f"mand_{i}", 
                        on_click=select_module, 
                        args=(module_name,),
                        use_container_width=True
                    )

            if recommended_modules:
                st.markdown("##### 🟡 Recommended Modules")
                cols = st.columns(3)
                for i, module_name in enumerate(recommended_modules):
                    cols[i % 3].button(
                        module_name, 
                        key=f"rec_{i}", 
                        on_click=select_module, 
                        args=(module_name,),
                        use_container_width=True
                    )
            st.markdown("---")

    # --- UNIFIED CHAT INPUT: ALWAYS ACTIVE ---
    if prompt := st.chat_input("Ask a question, continue with training, or chat about cybersecurity..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_user_input(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()