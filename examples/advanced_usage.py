"""
KAIROS Memory - Advanced Usage Example

Demonstrates:
- Large-scale memory storage
- Compression statistics
- Emotional encoding and retrieval
- Performance metrics
"""
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kairos import KAIROSMemory
import time


def main():
    print("=" * 70)
    print("KAIROS Memory - Advanced Usage Example")
    print("=" * 70)
    
    # Initialize memory system
    memory = KAIROSMemory(
        storage_path="./example_memory_advanced",
        use_multidim=True,
        enable_feedback=True
    )
    
    # Store a large dataset of conversation exchanges with emotions
    print("\n📝 Storing large dataset of conversation exchanges with emotional context...")
    
    # Generate a comprehensive dataset with 100+ exchanges
    exchanges = []
    
    # Food preferences (15 exchanges)
    food_exchanges = [
        ("I love pizza with pepperoni and extra cheese!", "That sounds delicious! Pizza is a great comfort food.", 0.6, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I'm craving sushi for dinner tonight.", "Sushi is a great choice! What's your favorite roll?", 0.5, {'emotion': 'ANTICIPATION', 'intensity': 0.6, 'pleasure': 0.5, 'arousal': 0.4}),
        ("I hate broccoli, it's so bitter!", "I understand. Some people find certain vegetables unappealing.", 0.4, {'emotion': 'DISGUST', 'intensity': 0.5, 'pleasure': -0.4, 'arousal': 0.3}),
        ("I enjoy Italian cuisine, especially pasta dishes.", "Italian food is known for its rich flavors and comforting qualities.", 0.6, {'emotion': 'CONTENT', 'intensity': 0.6, 'pleasure': 0.6, 'arousal': 0.2}),
        ("I love spicy food, the heat makes everything more exciting!", "Spicy food can be invigorating and add excitement to meals.", 0.5, {'emotion': 'EXCITEMENT', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.8}),
        ("I prefer vegetarian meals for health reasons.", "Vegetarian diets can be very nutritious and environmentally friendly.", 0.6, {'emotion': 'CALM', 'intensity': 0.5, 'pleasure': 0.4, 'arousal': 0.1}),
        ("I'm trying to eat healthier and cut down on sugar.", "That's a great goal! Reducing sugar can have many health benefits.", 0.7, {'emotion': 'DETERMINATION', 'intensity': 0.7, 'pleasure': 0.3, 'arousal': 0.5}),
        ("I love cooking at home, it's so therapeutic.", "Cooking can be a wonderful creative outlet and stress reliever.", 0.6, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.2}),
        ("I'm a big fan of Mexican food, especially tacos!", "Tacos are versatile and delicious. What's your favorite filling?", 0.5, {'emotion': 'JOY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.5}),
        ("I don't like seafood, the smell makes me nauseous.", "That's understandable. Food preferences are very personal.", 0.4, {'emotion': 'DISGUST', 'intensity': 0.6, 'pleasure': -0.5, 'arousal': 0.4}),
        ("I enjoy trying new restaurants and cuisines.", "Exploring different cuisines is a great way to experience new cultures.", 0.6, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.5}),
        ("I love breakfast food, especially pancakes and waffles.", "Breakfast foods are comforting and can be enjoyed any time of day.", 0.5, {'emotion': 'JOY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.3}),
        ("I'm on a keto diet and it's working well for me.", "Keto diets can be effective for some people. How do you feel on it?", 0.6, {'emotion': 'SATISFACTION', 'intensity': 0.6, 'pleasure': 0.6, 'arousal': 0.3}),
        ("I have a sweet tooth and love desserts.", "Desserts can be a wonderful treat in moderation.", 0.5, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.4}),
        ("I'm learning to bake and made my first loaf of bread!", "Baking is a rewarding skill. How did your bread turn out?", 0.7, {'emotion': 'PRIDE', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.6}),
    ]
    exchanges.extend(food_exchanges)
    
    # Colors and preferences (10 exchanges)
    color_exchanges = [
        ("My favorite color is blue, it reminds me of the ocean.", "Blue is a calming color, often associated with tranquility.", 0.5, {'emotion': 'CALM', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': -0.2}),
        ("I love the color red, it's so vibrant and energetic!", "Red is indeed a powerful color that can evoke strong emotions.", 0.6, {'emotion': 'EXCITEMENT', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.8}),
        ("Green is my favorite, it represents nature and growth.", "Green is associated with harmony, balance, and renewal.", 0.5, {'emotion': 'CALM', 'intensity': 0.6, 'pleasure': 0.5, 'arousal': 0.1}),
        ("I prefer neutral colors like beige and gray for my home.", "Neutral colors create a calming and sophisticated atmosphere.", 0.5, {'emotion': 'CALM', 'intensity': 0.5, 'pleasure': 0.4, 'arousal': -0.1}),
        ("Purple is such a royal and mysterious color!", "Purple is often associated with creativity, wisdom, and luxury.", 0.6, {'emotion': 'FASCINATION', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.4}),
        ("I love bright yellow, it's so cheerful and sunny!", "Yellow is associated with happiness, optimism, and energy.", 0.5, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.6}),
        ("Orange reminds me of autumn and warm sunsets.", "Orange combines the energy of red with the happiness of yellow.", 0.5, {'emotion': 'NOSTALGIA', 'intensity': 0.6, 'pleasure': 0.6, 'arousal': 0.3}),
        ("I prefer dark colors like navy and charcoal.", "Dark colors can create a sense of sophistication and depth.", 0.5, {'emotion': 'CALM', 'intensity': 0.5, 'pleasure': 0.4, 'arousal': -0.2}),
        ("Pink is so soft and gentle, I love it!", "Pink is often associated with compassion, love, and nurturing.", 0.5, {'emotion': 'LOVE', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.2}),
        ("I'm drawn to earth tones like brown and terracotta.", "Earth tones can create a warm, grounded feeling.", 0.5, {'emotion': 'CALM', 'intensity': 0.6, 'pleasure': 0.5, 'arousal': 0.0}),
    ]
    exchanges.extend(color_exchanges)
    
    # Technology and AI (20 exchanges)
    tech_exchanges = [
        ("Can you explain how neural networks work?", "Neural networks are computing systems inspired by biological neurons...", 0.9, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.4}),
        ("I'm learning about machine learning and it's fascinating!", "That's wonderful! Machine learning is a rapidly evolving field.", 0.8, {'emotion': 'EXCITEMENT', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I'm frustrated with my code not working.", "Debugging can be challenging. Let's break down the problem step by step.", 0.7, {'emotion': 'FRUSTRATION', 'intensity': 0.6, 'pleasure': -0.3, 'arousal': 0.7}),
        ("I just built my first web application!", "Congratulations! Building web apps is a great achievement.", 0.8, {'emotion': 'PRIDE', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.7}),
        ("I'm interested in learning Python programming.", "Python is a great language to start with. It's versatile and readable.", 0.7, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.5}),
        ("I love working with data science and analytics.", "Data science is a powerful field that combines statistics and programming.", 0.8, {'emotion': 'PASSION', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.5}),
        ("I'm confused about how APIs work.", "APIs allow different software applications to communicate with each other.", 0.7, {'emotion': 'CONFUSION', 'intensity': 0.5, 'pleasure': -0.2, 'arousal': 0.3}),
        ("I'm excited about the future of artificial intelligence!", "AI has the potential to transform many aspects of our lives.", 0.9, {'emotion': 'EXCITEMENT', 'intensity': 0.9, 'pleasure': 0.8, 'arousal': 0.8}),
        ("I'm worried about AI taking over jobs.", "That's a valid concern. It's important to adapt and learn new skills.", 0.7, {'emotion': 'ANXIETY', 'intensity': 0.7, 'pleasure': -0.4, 'arousal': 0.7}),
        ("I just fixed a major bug in my code!", "Great job! Debugging is an essential skill for developers.", 0.8, {'emotion': 'RELIEF', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': 0.5}),
        ("I'm learning about cloud computing and AWS.", "Cloud computing is essential for modern software development.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.5}),
        ("I love the challenge of solving complex algorithms.", "Algorithmic thinking is a valuable skill that improves with practice.", 0.8, {'emotion': 'SATISFACTION', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.6}),
        ("I'm struggling with understanding recursion.", "Recursion can be tricky at first. It helps to trace through examples step by step.", 0.7, {'emotion': 'FRUSTRATION', 'intensity': 0.6, 'pleasure': -0.2, 'arousal': 0.5}),
        ("I'm building a mobile app and it's going well!", "Mobile development is exciting. What platform are you targeting?", 0.8, {'emotion': 'EXCITEMENT', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.7}),
        ("I'm interested in cybersecurity and ethical hacking.", "Cybersecurity is crucial in our digital world. It's a growing field.", 0.8, {'emotion': 'INTEREST', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.5}),
        ("I just deployed my first application to production!", "That's a major milestone! How did the deployment go?", 0.9, {'emotion': 'PRIDE', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.8}),
        ("I'm learning about blockchain and cryptocurrency.", "Blockchain technology has many applications beyond cryptocurrency.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.5}),
        ("I'm overwhelmed by all the technologies I need to learn.", "It's normal to feel overwhelmed. Focus on one thing at a time.", 0.7, {'emotion': 'OVERWHELM', 'intensity': 0.7, 'pleasure': -0.3, 'arousal': 0.8}),
        ("I love contributing to open source projects!", "Open source contributions are a great way to learn and give back.", 0.8, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.5}),
        ("I'm studying computer science and really enjoying it.", "Computer science is a fascinating field with many opportunities.", 0.8, {'emotion': 'SATISFACTION', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.4}),
    ]
    exchanges.extend(tech_exchanges)
    
    # Personal experiences (15 exchanges)
    personal_exchanges = [
        ("I had an amazing vacation in Hawaii last summer!", "That sounds like a wonderful experience! What was your favorite part?", 0.8, {'emotion': 'JOY', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm feeling anxious about my upcoming presentation.", "It's normal to feel nervous. Preparation and practice can help build confidence.", 0.7, {'emotion': 'ANXIETY', 'intensity': 0.7, 'pleasure': -0.4, 'arousal': 0.8}),
        ("I'm so grateful for my supportive friends and family.", "Having a strong support system is truly valuable in life.", 0.8, {'emotion': 'GRATITUDE', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.3}),
        ("I just moved to a new city and I'm excited but nervous.", "Moving is a big change. It's normal to have mixed feelings.", 0.7, {'emotion': 'MIXED', 'intensity': 0.7, 'pleasure': 0.3, 'arousal': 0.6}),
        ("I'm celebrating my birthday this weekend!", "Happy birthday! I hope you have a wonderful celebration.", 0.7, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm feeling homesick after being away for months.", "Homesickness is natural. Staying connected with loved ones can help.", 0.6, {'emotion': 'SADNESS', 'intensity': 0.6, 'pleasure': -0.5, 'arousal': -0.2}),
        ("I just got a new pet and I'm so happy!", "Pets bring so much joy! What kind of pet did you get?", 0.8, {'emotion': 'JOY', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm worried about my health after recent symptoms.", "It's important to consult with healthcare professionals about concerns.", 0.8, {'emotion': 'ANXIETY', 'intensity': 0.8, 'pleasure': -0.5, 'arousal': 0.8}),
        ("I reconnected with an old friend and it was wonderful!", "Reconnecting with old friends can be very meaningful.", 0.7, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.5}),
        ("I'm feeling lonely lately despite being around people.", "Loneliness can happen even in crowds. It's about connection quality.", 0.7, {'emotion': 'LONELINESS', 'intensity': 0.7, 'pleasure': -0.4, 'arousal': -0.3}),
        ("I just finished reading an amazing book!", "What book was it? I'd love to hear about it.", 0.6, {'emotion': 'SATISFACTION', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.4}),
        ("I'm planning a trip to Europe next year.", "Europe has so much to offer! What countries are you most excited to visit?", 0.7, {'emotion': 'ANTICIPATION', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I'm dealing with a difficult family situation.", "Family issues can be challenging. It's important to communicate openly.", 0.8, {'emotion': 'STRESS', 'intensity': 0.7, 'pleasure': -0.4, 'arousal': 0.7}),
        ("I just achieved a personal goal I've been working toward!", "Congratulations! Achieving goals is incredibly rewarding.", 0.9, {'emotion': 'PRIDE', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.8}),
        ("I'm feeling overwhelmed by all my responsibilities.", "It's okay to feel overwhelmed. Breaking things into smaller tasks can help.", 0.7, {'emotion': 'OVERWHELM', 'intensity': 0.8, 'pleasure': -0.3, 'arousal': 0.8}),
    ]
    exchanges.extend(personal_exchanges)
    
    # Facts and knowledge (15 exchanges)
    knowledge_exchanges = [
        ("What's the capital of France?", "The capital of France is Paris, a beautiful city known for its art and culture.", 0.7, {'emotion': 'NEUTRAL', 'intensity': 0.3, 'pleasure': 0.0, 'arousal': 0.0}),
        ("Tell me about quantum computing.", "Quantum computing uses quantum mechanical phenomena like superposition...", 0.9, {'emotion': 'CURIOSITY', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': 0.5}),
        ("How does photosynthesis work?", "Photosynthesis is the process by which plants convert light energy into chemical energy.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': 0.4}),
        ("What is the theory of relativity?", "Einstein's theory of relativity describes how space and time are interconnected.", 0.9, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.5}),
        ("Can you explain how the internet works?", "The internet is a global network of interconnected computers using protocols like TCP/IP.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': 0.4}),
        ("What causes earthquakes?", "Earthquakes are caused by the movement of tectonic plates beneath the Earth's surface.", 0.7, {'emotion': 'CURIOSITY', 'intensity': 0.5, 'pleasure': 0.3, 'arousal': 0.3}),
        ("How do vaccines work?", "Vaccines train the immune system to recognize and fight specific pathogens.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': 0.4}),
        ("What is climate change?", "Climate change refers to long-term changes in global temperature and weather patterns.", 0.8, {'emotion': 'CONCERN', 'intensity': 0.6, 'pleasure': -0.2, 'arousal': 0.5}),
        ("How does the human brain process memories?", "Memory formation involves complex interactions between neurons and brain regions.", 0.9, {'emotion': 'FASCINATION', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.5}),
        ("What is dark matter?", "Dark matter is a hypothetical form of matter that doesn't interact with light.", 0.9, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.5}),
        ("How do black holes form?", "Black holes form when massive stars collapse under their own gravity.", 0.8, {'emotion': 'FASCINATION', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.5}),
        ("What is DNA and how does it work?", "DNA contains genetic instructions and uses a double helix structure.", 0.9, {'emotion': 'CURIOSITY', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.4}),
        ("How does evolution work?", "Evolution is the process by which species change over time through natural selection.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': 0.4}),
        ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second.", 0.7, {'emotion': 'NEUTRAL', 'intensity': 0.4, 'pleasure': 0.2, 'arousal': 0.2}),
        ("How do solar panels convert sunlight into electricity?", "Solar panels use photovoltaic cells to convert light energy into electrical energy.", 0.8, {'emotion': 'CURIOSITY', 'intensity': 0.6, 'pleasure': 0.4, 'arousal': 0.4}),
    ]
    exchanges.extend(knowledge_exchanges)
    
    # Hobbies and interests (15 exchanges)
    hobby_exchanges = [
        ("I enjoy reading science fiction novels in my spare time.", "Science fiction is a great genre that explores fascinating possibilities.", 0.6, {'emotion': 'CONTENT', 'intensity': 0.6, 'pleasure': 0.6, 'arousal': 0.2}),
        ("I love playing guitar, it helps me relax after a long day.", "Music is a wonderful way to unwind and express creativity.", 0.7, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.1}),
        ("I'm passionate about photography and love capturing nature scenes.", "Photography is a beautiful art form that allows you to preserve special moments.", 0.7, {'emotion': 'LOVE', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.4}),
        ("I love hiking and exploring nature trails.", "Hiking is great exercise and a wonderful way to connect with nature.", 0.7, {'emotion': 'JOY', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.5}),
        ("I enjoy painting as a creative outlet.", "Painting is a wonderful form of self-expression and relaxation.", 0.6, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.2}),
        ("I'm learning to play chess and it's challenging!", "Chess is a great game that exercises strategic thinking.", 0.7, {'emotion': 'DETERMINATION', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.5}),
        ("I love gardening and watching my plants grow.", "Gardening is rewarding and can be very therapeutic.", 0.6, {'emotion': 'CONTENT', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.2}),
        ("I enjoy playing video games with friends online.", "Gaming can be a great way to socialize and have fun.", 0.6, {'emotion': 'JOY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I'm into fitness and go to the gym regularly.", "Regular exercise has many physical and mental health benefits.", 0.7, {'emotion': 'ENERGY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I love cooking and trying new recipes.", "Cooking is both creative and practical. What's your favorite dish to make?", 0.6, {'emotion': 'JOY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.3}),
        ("I enjoy writing poetry in my journal.", "Poetry is a beautiful way to express emotions and thoughts.", 0.6, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.1}),
        ("I'm learning a new language and it's exciting!", "Learning languages opens up new cultures and opportunities.", 0.7, {'emotion': 'EXCITEMENT', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I love dancing, it's so freeing and fun!", "Dancing is a great form of exercise and self-expression.", 0.6, {'emotion': 'JOY', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.8}),
        ("I enjoy woodworking and building things with my hands.", "Woodworking is a satisfying craft that combines skill and creativity.", 0.7, {'emotion': 'SATISFACTION', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.4}),
        ("I'm passionate about bird watching and nature photography.", "Bird watching requires patience and observation skills.", 0.6, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.2}),
    ]
    exchanges.extend(hobby_exchanges)
    
    # Work and career (15 exchanges)
    work_exchanges = [
        ("I just got promoted at work and I'm so excited!", "Congratulations! That's a significant achievement worth celebrating.", 0.9, {'emotion': 'PRIDE', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm stressed about meeting my project deadline.", "Time management and breaking tasks into smaller steps can help reduce stress.", 0.7, {'emotion': 'STRESS', 'intensity': 0.7, 'pleasure': -0.5, 'arousal': 0.8}),
        ("I love my job and feel fulfilled by my work.", "It's wonderful when work aligns with your passions and values.", 0.8, {'emotion': 'SATISFACTION', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.4}),
        ("I'm looking for a new job and feeling anxious about interviews.", "Job searching can be stressful. Preparation and practice can help build confidence.", 0.8, {'emotion': 'ANXIETY', 'intensity': 0.7, 'pleasure': -0.3, 'arousal': 0.7}),
        ("I just completed a major project successfully!", "That's a great accomplishment! How do you feel about the results?", 0.9, {'emotion': 'PRIDE', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm feeling burned out from work lately.", "Burnout is serious. It's important to take breaks and prioritize self-care.", 0.8, {'emotion': 'EXHAUSTION', 'intensity': 0.8, 'pleasure': -0.4, 'arousal': -0.3}),
        ("I'm excited about a new project I'm starting!", "New projects can be energizing. What makes this one special?", 0.8, {'emotion': 'EXCITEMENT', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.7}),
        ("I'm struggling with work-life balance.", "Work-life balance is challenging but important for well-being.", 0.7, {'emotion': 'STRESS', 'intensity': 0.7, 'pleasure': -0.4, 'arousal': 0.7}),
        ("I just received positive feedback from my manager!", "That's wonderful! Recognition for your work is always encouraging.", 0.8, {'emotion': 'PRIDE', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.5}),
        ("I'm considering a career change but I'm nervous.", "Career changes are big decisions. What's drawing you to make the change?", 0.8, {'emotion': 'ANXIETY', 'intensity': 0.7, 'pleasure': 0.2, 'arousal': 0.6}),
        ("I love collaborating with my team on projects.", "Team collaboration can lead to great results and build relationships.", 0.7, {'emotion': 'JOY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.4}),
        ("I'm frustrated with office politics and drama.", "Workplace dynamics can be challenging. Focus on what you can control.", 0.7, {'emotion': 'FRUSTRATION', 'intensity': 0.7, 'pleasure': -0.4, 'arousal': 0.7}),
        ("I just got a raise and I'm thrilled!", "Congratulations! A raise is great recognition of your contributions.", 0.9, {'emotion': 'JOY', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm learning new skills for my career development.", "Continuous learning is key to career growth. What skills are you focusing on?", 0.8, {'emotion': 'DETERMINATION', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I'm feeling undervalued at my current job.", "It's important to feel valued. Have you discussed this with your manager?", 0.7, {'emotion': 'DISAPPOINTMENT', 'intensity': 0.7, 'pleasure': -0.5, 'arousal': 0.4}),
    ]
    exchanges.extend(work_exchanges)
    
    # Health and wellness (15 exchanges)
    health_exchanges = [
        ("I've been feeling really tired lately.", "It's important to get enough rest. Consider your sleep schedule and stress levels.", 0.6, {'emotion': 'FATIGUE', 'intensity': 0.6, 'pleasure': -0.3, 'arousal': -0.4}),
        ("I started a new exercise routine and I'm feeling great!", "Regular exercise has many benefits for both physical and mental health.", 0.7, {'emotion': 'ENERGY', 'intensity': 0.7, 'pleasure': 0.7, 'arousal': 0.6}),
        ("I'm trying to improve my sleep schedule.", "Good sleep is crucial for health. What changes are you making?", 0.7, {'emotion': 'DETERMINATION', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.4}),
        ("I've been meditating daily and it's helping my stress.", "Meditation is a powerful tool for managing stress and improving well-being.", 0.7, {'emotion': 'CALM', 'intensity': 0.8, 'pleasure': 0.7, 'arousal': -0.2}),
        ("I'm struggling with anxiety and it's affecting my daily life.", "Anxiety can be challenging. Have you considered speaking with a professional?", 0.8, {'emotion': 'ANXIETY', 'intensity': 0.8, 'pleasure': -0.5, 'arousal': 0.8}),
        ("I just completed a 5K run and I'm proud of myself!", "That's a great achievement! Running is excellent exercise.", 0.8, {'emotion': 'PRIDE', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm trying to eat more mindfully and listen to my body.", "Mindful eating can improve your relationship with food.", 0.7, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.1}),
        ("I've been having trouble sleeping through the night.", "Sleep issues can have many causes. Consider your routine and environment.", 0.7, {'emotion': 'FRUSTRATION', 'intensity': 0.6, 'pleasure': -0.3, 'arousal': 0.4}),
        ("I started doing yoga and it's been transformative!", "Yoga combines physical movement with mindfulness. How do you feel?", 0.7, {'emotion': 'CALM', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.2}),
        ("I'm feeling more energetic after changing my diet.", "Diet can have a significant impact on energy levels. What changes did you make?", 0.7, {'emotion': 'ENERGY', 'intensity': 0.8, 'pleasure': 0.8, 'arousal': 0.6}),
        ("I'm dealing with chronic pain and it's frustrating.", "Chronic pain is challenging. Have you explored different management strategies?", 0.8, {'emotion': 'FRUSTRATION', 'intensity': 0.7, 'pleasure': -0.5, 'arousal': 0.5}),
        ("I just hit my fitness goal and I'm so happy!", "Congratulations! Achieving fitness goals takes dedication and consistency.", 0.9, {'emotion': 'JOY', 'intensity': 0.9, 'pleasure': 0.9, 'arousal': 0.7}),
        ("I'm trying to reduce my screen time for better sleep.", "Reducing screen time, especially before bed, can improve sleep quality.", 0.7, {'emotion': 'DETERMINATION', 'intensity': 0.7, 'pleasure': 0.5, 'arousal': 0.3}),
        ("I've been feeling more balanced since starting therapy.", "Therapy can be very helpful for mental health and personal growth.", 0.8, {'emotion': 'HOPE', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': 0.2}),
        ("I'm learning to manage my stress better through breathing exercises.", "Breathing exercises are a simple but effective stress management tool.", 0.7, {'emotion': 'CALM', 'intensity': 0.7, 'pleasure': 0.6, 'arousal': -0.1}),
    ]
    exchanges.extend(health_exchanges)
    
    stored_count = 0
    total_original_size = 0
    total_compressed_size = 0
    
    for user_msg, assistant_msg, importance, emotion_meta in exchanges:
        token_id = memory.consolidate_exchange(
            user_msg, 
            assistant_msg, 
            importance,
            metadata=emotion_meta
        )
        if token_id:
            stored_count += 1
            exchange_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
            total_original_size += len(exchange_text.encode('utf-8'))
            print(f"  ✓ Stored [{emotion_meta.get('emotion', 'NEUTRAL')}]: '{user_msg[:50]}...'")
    
    print(f"\n  Total stored: {stored_count} exchanges")
    
    # Retrieve relevant memories
    print("\n🔍 Retrieving relevant memories...")
    
    test_queries = [
        ("What food do I like?", "Food preferences"),
        ("Tell me about AI and technology", "Technology interests"),
        ("What's my favorite color?", "Color preferences"),
        ("How am I feeling emotionally?", "Emotional state"),
        ("What are my hobbies?", "Hobbies and interests"),
    ]
    
    for query, category in test_queries:
        print(f"\n  Query [{category}]: '{query}'")
        results, session_id = memory.retrieve_relevant(query, top_k=3)
        
        if results:
            for i, mem in enumerate(results, 1):
                emotion = mem['metadata'].get('emotion', 'NEUTRAL')
                intensity = mem['metadata'].get('intensity', 0.0)
                print(f"    {i}. (sim={mem['similarity']:.3f}, {emotion}={intensity:.1f}) {mem['content'][:55]}...")
        else:
            print("    No results found")
    
    # Emotional retrieval
    print("\n😊 Emotional Memory Retrieval:")
    emotions_to_test = ['JOY', 'EXCITEMENT', 'ANXIETY', 'CALM']
    for emotion in emotions_to_test:
        results = memory.retrieve_by_emotion(emotion, top_k=2)
        if results:
            print(f"\n  {emotion} memories:")
            for i, mem in enumerate(results, 1):
                intensity = mem.get('intensity', 0.0)
                print(f"    {i}. (intensity={intensity:.2f}) {mem['content'][:55]}...")
        else:
            print(f"\n  {emotion}: No memories found")
    
    # Show comprehensive statistics
    print("\n📊 Comprehensive Statistics:")
    print("=" * 70)
    stats = memory.get_stats()
    
    print("\n📈 Storage Metrics:")
    print(f"  • Total documents stored: {stats['total_documents']}")
    print(f"  • Total stores: {stats['stores']}")
    print(f"  • Total retrievals: {stats['retrievals']}")
    print(f"  • Total characters stored: {stats['total_chars_stored']:,}")
    print(f"  • Total bytes stored: {stats['total_bytes_stored']:,}")
    print(f"  • Storage size: {stats['storage_size_mb']:.3f} MB")
    
    print("\n⚡ Performance Metrics:")
    store_lat = stats['store_latency_ms']
    retrieve_lat = stats['retrieve_latency_ms']
    print(f"  • Store latency:")
    print(f"    - Mean: {store_lat['mean']:.2f}ms")
    print(f"    - P50: {store_lat['p50']:.2f}ms")
    print(f"    - P95: {store_lat['p95']:.2f}ms")
    print(f"  • Retrieve latency:")
    print(f"    - Mean: {retrieve_lat['mean']:.2f}ms")
    print(f"    - P50: {retrieve_lat['p50']:.2f}ms")
    print(f"    - P95: {retrieve_lat['p95']:.2f}ms")
    
    print("\n🎯 Quality Metrics:")
    print(f"  • Average similarity score: {stats['avg_similarity_score']:.3f}")
    print(f"  • Feedback sessions: {stats['feedback_sessions']}")
    print(f"  • Multi-dimensional encoding: {'Enabled' if stats['multidim_enabled'] else 'Disabled'}")
    print(f"  • Feedback loop: {'Enabled' if stats['feedback_enabled'] else 'Disabled'}")
    
    # Compression statistics
    print("\n💾 Compression Statistics:")
    if stats['total_bytes_stored'] > 0 and stats['total_documents'] > 0:
        # Estimate vector storage size
        # Each 408d vector = 408 * 4 bytes = 1632 bytes (float32)
        # Each 384d vector = 384 * 4 bytes = 1536 bytes (float32)
        vector_size_per_doc = 1632 if stats['multidim_enabled'] else 1536
        estimated_vector_size = stats['total_documents'] * vector_size_per_doc
        
        # Original text size
        original_size = stats['total_bytes_stored']
        avg_text_per_doc = original_size / stats['total_documents']
        
        # Compression ratio (original / compressed)
        compression_ratio = original_size / estimated_vector_size if estimated_vector_size > 0 else 0
        
        # Space efficiency
        if original_size > estimated_vector_size:
            space_savings = ((1 - estimated_vector_size / original_size) * 100)
            efficiency = "Compressed"
        else:
            space_savings = ((estimated_vector_size / original_size - 1) * 100)
            efficiency = "Expanded"
        
        print(f"  • Original text size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
        print(f"  • Average text per document: {avg_text_per_doc:.0f} bytes")
        print(f"  • Vector storage size: ~{estimated_vector_size:,} bytes ({estimated_vector_size/1024:.2f} KB)")
        print(f"  • Vector size per document: {vector_size_per_doc} bytes (fixed)")
        print(f"  • Compression ratio: {compression_ratio:.2f}x ({efficiency})")
        if compression_ratio > 1:
            print(f"  • Space savings: {space_savings:.1f}%")
        else:
            print(f"  • Space overhead: {space_savings:.1f}% (vectors are fixed-size)")
        print(f"  • Note: Vectors enable semantic search; compression benefits increase with longer texts")
    
    # Emotion statistics
    print("\n😊 Emotion Statistics:")
    all_tokens = memory.latent_store.get_all_tokens()
    emotion_counts = {}
    emotion_intensities = {}
    
    for token_id, data in all_tokens.items():
        meta = data.get('metadata', {})
        emotion = meta.get('emotion', 'NEUTRAL')
        intensity = meta.get('intensity', 0.0)
        
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
            emotion_intensities[emotion] = []
        emotion_counts[emotion] += 1
        emotion_intensities[emotion].append(intensity)
    
    if emotion_counts:
        print(f"  • Total emotions tracked: {len(emotion_counts)}")
        print(f"  • Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            avg_intensity = sum(emotion_intensities[emotion]) / len(emotion_intensities[emotion])
            percentage = (count / len(all_tokens)) * 100
            print(f"    - {emotion:15s}: {count:2d} memories ({percentage:5.1f}%), avg intensity: {avg_intensity:.2f}")
    else:
        print("  • No emotion data available")
    
    print("\n✅ Advanced example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
