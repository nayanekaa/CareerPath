from typing import List, Dict

QUESTIONS = [
    {
        'id': 1,
        'text': 'When facing a completely new problem, what is your first instinct?',
        'options': ['theory','practical','visual','social']
    },
    {
        'id': 2,
        'text': 'In a team setting, which role do you naturally gravitate towards?',
        'options': ['structure','deep_work','connector','driver']
    },
    {
        'id': 3,
        'text': 'How do you prefer to learn a new technology?',
        'options': ['docs','video','project']
    },
    {
        'id': 4,
        'text': 'What motivates you most in a career?',
        'options': ['mastery','creation','logic']
    }
]

def score_responses(responses: List[Dict]) -> Dict:
    """responses: list of {id: int, answer: str}
    returns a psychometric profile mapping to archetypes and learning/work styles.
    """
    traits = []
    learningStyle = 'Practical'
    workStyle = 'Independent'

    for r in responses:
        ans = r.get('answer')
        if ans in ('theory','docs'):
            learningStyle = 'Theoretical'
            traits.append('Analytical')
        if ans in ('practical','project'):
            learningStyle = 'Practical'
            traits.append('Pragmatic')
        if ans in ('visual','video'):
            learningStyle = 'Visual'
            traits.append('Visualizer')
        if ans in ('social','connector'):
            workStyle = 'Collaborative'
            traits.append('Collaborative')
        if ans == 'structure':
            workStyle = 'Structured'
            traits.append('Organized')
        if ans == 'driver':
            traits.append('Results-Oriented')
        if ans == 'deep_work':
            workStyle = 'Independent'
            traits.append('Focused')
        if ans == 'mastery':
            traits.append('Perfectionist')
        if ans == 'creation':
            traits.append('Creative')
        if ans == 'logic':
            traits.append('Logical')

    # derive archetype
    archetype = 'Versatile Generalist'
    if 'Analytical' in traits and 'Perfectionist' in traits:
        archetype = 'Strategic Architect'
    if 'Pragmatic' in traits and 'Builder' in traits:
        archetype = 'Pragmatic Builder'
    if 'Visualizer' in traits and 'Creative' in traits:
        archetype = 'Creative Visionary'
    if 'Collaborative' in traits and 'Empathetic' in traits:
        archetype = 'Team Catalyst'

    return {
        'archetype': archetype,
        'learningStyle': learningStyle,
        'workStyle': workStyle,
        'traits': traits,
        'description': f'{archetype} who learns best via {learningStyle} methods.'
    }
