

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