"""Generate synthetic Q&A pairs for LoRA fine-tuning."""

import json
import random
from typing import List, Dict

# Sample career-related Q&A pairs for fine-tuning
SAMPLE_QA_PAIRS = [
    {
        "question": "What programming languages should a junior backend engineer learn?",
        "answer": "A junior backend engineer should focus on Python, JavaScript (Node.js), or Java. Python is beginner-friendly with frameworks like Django/FastAPI. Java is industry-standard with Spring Boot. JavaScript with Node.js is popular for full-stack work. Start with one deeply, then learn another to understand different paradigms."
    },
    {
        "question": "How do I transition from frontend to backend development?",
        "answer": "Start by learning backend fundamentals: databases (SQL, NoSQL), APIs (REST, GraphQL), and a backend language. Build 2-3 projects with growing complexity. Study system design and architecture. Contribute to open-source backend projects. Network with backend engineers. Most importantly, show real-world backend projects in your portfolio."
    },
    {
        "question": "What skills are essential for a DevOps engineer?",
        "answer": "Core DevOps skills: Linux administration, containerization (Docker), orchestration (Kubernetes), CI/CD pipelines (Jenkins, GitLab CI), Infrastructure as Code (Terraform, Ansible), cloud platforms (AWS, GCP, Azure), monitoring tools (Prometheus, Grafana), and scripting (Bash, Python)."
    },
    {
        "question": "How can I prepare for a technical interview?",
        "answer": "Practice coding problems on LeetCode (50+ medium problems minimum). Study system design patterns and trade-offs. Understand data structures and algorithms deeply. Review your own projects and be ready to discuss technical decisions. Practice mock interviews. On interview day: clarify requirements, think aloud, and optimize step-by-step."
    },
    {
        "question": "What is the learning curve for cloud platforms like AWS?",
        "answer": "AWS has a steep initial learning curve (1-3 months) due to service variety (200+ services), but focus on core services first: EC2, S3, RDS, Lambda, CloudFormation. Use free tier and hands-on projects. AWS certifications (Solutions Architect Associate) can validate knowledge. Budget 3-6 months to become proficient."
    },
    {
        "question": "How do I build a portfolio project that impresses employers?",
        "answer": "Choose a problem you care about. Use modern tech stack. Focus on code quality, not features count. Include: clear README, clean architecture, error handling, tests, and deployment. Open-source contributions are great. Blog about your learning journey. Showcase on GitHub with detailed commit history."
    },
    {
        "question": "What's the difference between SQL and NoSQL databases?",
        "answer": "SQL (PostgreSQL, MySQL): structured data, ACID guarantees, joins, best for relational data. NoSQL (MongoDB, DynamoDB): flexible schema, horizontal scaling, best for unstructured data. Use SQL for financial/transactional systems. Use NoSQL for user profiles, logs, real-time data. Many projects use both."
    },
    {
        "question": "How can I improve my problem-solving skills?",
        "answer": "Practice daily: solve algorithmic problems, break down complex problems into smaller parts, understand edge cases. Study others' solutions. Teach others (blog/YouTube). Work on real projects. Don't just memorize—understand the 'why'. Keep a learning journal of patterns and techniques."
    },
    {
        "question": "What's the best way to learn a new programming language?",
        "answer": "Build projects immediately—don't just read syntax. Start small (CLI tools, scripts). Focus on core concepts: variables, functions, OOP, error handling. Use documentation and Stack Overflow. Join communities (Reddit, Discord). The 80/20 rule: learn 20% of syntax to do 80% of work."
    },
    {
        "question": "How do I transition careers into tech?",
        "answer": "Start with fundamentals (online courses: Udemy, Coursera, freeCodeCamp). Build 3-5 portfolio projects. Network: attend meetups, tech talks, online communities. Get a foothold: internship, contract work, or entry-level role. Be patient—6-12 months is realistic. Consider bootcamps if you need structure."
    },
]


def generate_qa_pairs(n: int = 50, seed: int = None) -> List[Dict[str, str]]:
    """
    Generate synthetic Q&A pairs for LoRA fine-tuning.
    
    This expands sample pairs with variations and generates new ones
    by combining related topics.
    
    Args:
        n: Number of pairs to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of Q&A dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    pairs = []
    
    # Add base pairs
    pairs.extend(SAMPLE_QA_PAIRS)
    
    # Generate variations and combinations
    if n > len(SAMPLE_QA_PAIRS):
        for i in range(n - len(SAMPLE_QA_PAIRS)):
            # Randomly select a base pair and create variation
            base_pair = random.choice(SAMPLE_QA_PAIRS)
            
            # Create variations by modifying the question
            variations = [
                f"What are the best practices for {base_pair['question'].split('for ')[-1] if 'for' in base_pair['question'] else 'software development'}?",
                f"How can I improve in {base_pair['question'].lower().split('?')[0].split()[-1]}?",
                f"What resources would you recommend for learning {base_pair['question'].lower().split()[-2:]}?",
            ]
            
            if i < len(variations):
                pairs.append({
                    "question": variations[i % len(variations)],
                    "answer": base_pair["answer"]
                })
            else:
                # Add the original pair again with slight modifications
                modified_answer = f"{base_pair['answer'][:200]}... You can learn more by practicing and building projects."
                pairs.append({
                    "question": base_pair["question"],
                    "answer": modified_answer
                })
    
    return pairs[:n]


def format_for_training(qa_pairs: List[Dict[str, str]]) -> List[str]:
    """
    Format Q&A pairs for language model training.
    
    Args:
        qa_pairs: List of Q&A dictionaries
    
    Returns:
        List of formatted prompt-completion pairs
    """
    formatted = []
    for pair in qa_pairs:
        # Format as: "Q: {question}\nA: {answer}"
        prompt = f"Q: {pair['question']}\nA:"
        completion = f" {pair['answer']}"
        formatted.append(f"{prompt}{completion}")
    
    return formatted


def save_qa_pairs(qa_pairs: List[Dict[str, str]], filepath: str):
    """Save Q&A pairs to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)


def load_qa_pairs(filepath: str) -> List[Dict[str, str]]:
    """Load Q&A pairs from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate and save sample dataset
    pairs = generate_qa_pairs(n=50, seed=42)
    print(f"Generated {len(pairs)} Q&A pairs")
    
    # Save to file
    import os
    dataset_dir = os.path.dirname(__file__)
    save_qa_pairs(pairs, os.path.join(dataset_dir, "lora_qa_dataset.json"))
    print(f"Saved to {dataset_dir}/lora_qa_dataset.json")
    
    # Show sample
    print("\nSample pairs:")
    for i, pair in enumerate(pairs[:3]):
        print(f"{i+1}. Q: {pair['question'][:60]}...")
        print(f"   A: {pair['answer'][:80]}...\n")
