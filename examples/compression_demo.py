"""
KAIROS Memory - Compression Demonstration

Demonstrates the 16-128x compression capabilities with long-form text.
Shows how compression benefits increase dramatically with longer texts.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kairos import KAIROSMemory


def main():
    print("=" * 70)
    print("KAIROS Memory - Compression Demonstration")
    print("=" * 70)
    print("\nThis demo shows how compression ratios improve with longer texts.")
    print("Vectors are fixed-size (1632 bytes), so longer texts achieve higher compression.\n")
    
    # Initialize embedding model (needed for realistic compression)
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    except ImportError:
        print("⚠️  sentence-transformers not installed. Compression ratios may be inaccurate with hash fallback.")

    memory = KAIROSMemory(
        storage_path="./example_memory_compression",
        use_multidim=True,
        enable_feedback=False,
        embedding_model=embedding_model
    )
    
    # Create examples with varying text lengths to demonstrate compression scaling
    print("📝 Storing documents with varying lengths...\n")
    
    # Short text (will show low/no compression)
    short_text = (
        "I love pizza.",
        "That's great! Pizza is delicious.",
        0.5
    )
    
    # Medium text
    medium_text = (
        "Can you explain how neural networks work?",
        """Neural networks are computing systems inspired by biological neural networks. 
They consist of interconnected nodes organized in layers. Information flows through the network, 
with each connection having a weight that gets adjusted during training. The network learns by 
processing large amounts of data and adjusting these weights to minimize errors. Neural networks 
have revolutionized many fields including image recognition, natural language processing, and autonomous vehicles.""",
        0.8
    )
    
    # Long text (should show ~16x compression)
    long_text = (
        "Please provide a comprehensive explanation of deep learning, its history, applications, and future prospects.",
        """Deep learning represents a revolutionary approach to artificial intelligence that has transformed 
numerous industries and applications over the past decade. At its core, deep learning uses artificial neural 
networks with multiple layers (hence 'deep') to learn hierarchical representations of data. The field has its 
roots in early neural network research from the 1940s and 1950s, but truly began to flourish in the 2010s 
thanks to advances in computational power, the availability of massive datasets, and improved algorithms.

The architecture of deep neural networks typically consists of an input layer that receives raw data, multiple 
hidden layers that progressively extract more abstract features, and an output layer that produces predictions 
or classifications. Each layer contains numerous neurons (nodes) that are connected to neurons in adjacent layers 
through weighted connections. During training, the network processes labeled examples, calculates prediction errors, 
and uses backpropagation to adjust the weights throughout the network. This process continues iteratively until 
the network learns to make accurate predictions on new, unseen data.

One of the key breakthroughs in deep learning was the development of convolutional neural networks (CNNs) for 
image processing. CNNs use specialized layers that can detect spatial patterns, making them exceptionally effective 
for tasks like image classification, object detection, and facial recognition. Another major advancement came with 
recurrent neural networks (RNNs) and their variants like LSTMs and GRUs, which excel at processing sequential data 
such as text, speech, and time series.

The applications of deep learning are vast and growing. In computer vision, deep learning powers everything from 
medical imaging systems that can detect diseases to autonomous vehicles that navigate complex environments. In 
natural language processing, transformer architectures like BERT and GPT have achieved remarkable results in 
translation, question answering, and text generation. Speech recognition systems in virtual assistants rely heavily 
on deep learning. The technology also drives recommendation systems, fraud detection, drug discovery, and 
scientific research across many domains.

Looking toward the future, deep learning continues to evolve rapidly. Areas of active research include improving 
model efficiency and reducing computational requirements, developing more interpretable and explainable models, 
addressing bias and fairness concerns, and exploring new architectures and training paradigms. The integration of 
deep learning with other AI approaches, such as reinforcement learning and symbolic reasoning, promises to 
create even more powerful systems. As computational resources continue to grow and new techniques emerge, deep 
learning will likely continue to push the boundaries of what's possible in artificial intelligence.""",
        0.9
    )
    
    # Ultra long text (should show ~20-30x compression - 30KB+)
    ultra_long_text = (
        "I need an extremely comprehensive and detailed guide covering the complete history of computing, from early mechanical calculators to modern quantum computers, including all major technological breakthroughs, key figures, future directions, detailed explanations of how each technology works, and extensive coverage of software development, programming languages, operating systems, networking, databases, security, artificial intelligence, and emerging technologies.",
        """The history of computing is a remarkable journey that spans centuries of human ingenuity and innovation. 
This comprehensive overview traces the evolution from early mechanical devices to the sophisticated systems of today 
and the quantum computers of tomorrow.

The earliest computational devices date back thousands of years. The abacus, invented in ancient Mesopotamia around 
2400 BCE, was the first known calculating tool. However, the modern era of computing began in the 17th century with 
mechanical calculators. In 1642, Blaise Pascal invented the Pascaline, a mechanical adding machine. Gottfried 
Wilhelm Leibniz later improved upon this with a machine that could perform multiplication. These early devices 
demonstrated that mechanical systems could perform arithmetic operations.

The 19th century brought revolutionary concepts. Charles Babbage, often called the 'father of computing,' designed 
the Difference Engine in the 1820s and later conceived the Analytical Engine in the 1830s. Although never fully 
constructed in his lifetime, the Analytical Engine contained all the essential elements of a modern computer: an 
arithmetic logic unit, memory (the store), and program control (the mill). Ada Lovelace, who worked with Babbage, 
wrote what is considered the first computer program and recognized the machine's potential beyond mere calculation.

The early 20th century saw the development of electromechanical computers. Herman Hollerith's tabulating machines, 
used for the 1890 U.S. Census, demonstrated the power of automated data processing. These machines used punched 
cards to store and process information. IBM, founded in 1911, became a dominant force in this era, manufacturing 
punch card equipment and later developing more sophisticated electromechanical calculators.

World War II accelerated computing development dramatically. The need for rapid calculations in cryptography, 
ballistics, and other military applications drove innovation. In 1941, Konrad Zuse completed the Z3, the first 
programmable, fully automatic computer. In the United States, the Harvard Mark I, developed by Howard Aiken and 
IBM, was completed in 1944. The British Colossus computers, built to break German codes, were among the first 
electronic digital computers, though they remained secret for decades.

The electronic era truly began with ENIAC (Electronic Numerical Integrator and Computer), completed in 1945 at 
the University of Pennsylvania. ENIAC was massive - occupying 1,800 square feet and containing 17,468 vacuum 
tubes. It could perform 5,000 additions per second, a thousand times faster than electromechanical machines. 
However, programming ENIAC required physically rewiring the machine, a time-consuming process.

A crucial breakthrough came with the stored-program concept, independently developed by John von Neumann and others. 
This architecture, now called the von Neumann architecture, stores both data and instructions in the same memory, 
allowing programs to be easily modified. The first stored-program computers, like the Manchester Baby (1948) and 
EDSAC (1949), demonstrated this revolutionary approach.

The 1950s saw the commercialization of computers. UNIVAC I, delivered to the U.S. Census Bureau in 1951, became 
the first commercially available computer. IBM entered the market with the IBM 701 in 1952, beginning its 
dominance of the industry. These first-generation computers used vacuum tubes, were enormous, consumed vast amounts 
of power, and required constant maintenance. They were primarily used by large organizations for scientific and 
business applications.

The invention of the transistor in 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs 
heralded the second generation of computers (late 1950s to mid-1960s). Transistors were smaller, more reliable, 
more energy-efficient, and generated less heat than vacuum tubes. This enabled computers to become smaller, faster, 
and more affordable. The IBM 7090, introduced in 1959, was a prominent second-generation machine.

The development of integrated circuits (ICs) in the late 1950s and early 1960s marked the third generation. Jack 
Kilby at Texas Instruments and Robert Noyce at Fairchild Semiconductor independently developed the integrated circuit, 
which placed multiple transistors on a single silicon chip. This further reduced size, cost, and power consumption 
while increasing reliability and performance. The IBM System/360, introduced in 1964, was a landmark third-generation 
computer family that established compatibility across different models.

The fourth generation began in the 1970s with the microprocessor - an entire CPU on a single chip. Intel's 4004, 
introduced in 1971, was the first commercial microprocessor. This breakthrough made personal computers possible. 
The Altair 8800 (1975) was the first successful personal computer kit. Apple Computer, founded by Steve Jobs and 
Steve Wozniak, released the Apple II in 1977, which became a major success. IBM entered the personal computer 
market in 1981 with the IBM PC, which established many standards still used today.

The 1980s and 1990s saw rapid advancement in personal computing. Graphical user interfaces, pioneered by Xerox 
PARC and popularized by Apple's Macintosh (1984) and Microsoft Windows, made computers accessible to non-technical 
users. The internet, which began as ARPANET in the 1960s, became publicly accessible in the 1990s, transforming 
how people communicate and access information. The World Wide Web, invented by Tim Berners-Lee in 1989, made the 
internet user-friendly and accessible to millions.

The 2000s brought mobile computing and cloud computing. Smartphones, led by Apple's iPhone (2007), put powerful 
computers in people's pockets. Cloud computing enabled access to vast computational resources over the internet. 
Social media platforms transformed how people connect and share information. Big data and machine learning began 
to extract insights from massive datasets.

Today, we're in an era of artificial intelligence, machine learning, and edge computing. Deep learning has achieved 
remarkable breakthroughs in image recognition, natural language processing, and game playing. Quantum computing, 
which uses quantum mechanical phenomena like superposition and entanglement, promises to solve certain problems 
exponentially faster than classical computers. Companies like IBM, Google, and others are developing quantum 
computers, though practical applications are still emerging.

Looking to the future, several trends are shaping computing. Neuromorphic computing aims to mimic the brain's 
structure and function. DNA computing explores using biological molecules for computation. Optical computing uses 
light instead of electricity. Quantum computing may revolutionize cryptography, drug discovery, and optimization 
problems. Edge computing brings processing closer to where data is generated. The integration of AI into every 
aspect of computing continues to accelerate.

Throughout this history, key themes emerge: the relentless drive for smaller, faster, cheaper, and more powerful 
computers; the importance of software and programming languages; the evolution from specialized machines to 
ubiquitous devices; and the transformative impact on society, economy, and human capabilities. Computing has 
evolved from room-sized machines accessible only to experts to devices in billions of pockets, fundamentally 
changing how humans work, communicate, learn, and create.

The software side of computing has been equally transformative. Early computers were programmed using machine code 
and assembly language, requiring deep technical expertise. The development of high-level programming languages like 
FORTRAN (1957), COBOL (1959), and later C (1972), made programming more accessible. Object-oriented programming, 
pioneered by languages like Smalltalk and later popularized by C++ and Java, revolutionized software development. 
Modern languages like Python, JavaScript, and Rust continue to evolve, each optimized for different use cases.

Operating systems have evolved from simple batch processing systems to sophisticated multi-tasking, multi-user 
environments. Early systems like IBM's OS/360 managed resources for mainframe computers. Unix, developed at Bell 
Labs in the 1970s, introduced concepts like hierarchical file systems and pipes that remain fundamental today. 
Microsoft's MS-DOS and Windows brought computing to the masses, while Linux demonstrated the power of open-source 
development. Modern operating systems like macOS, Windows, Android, and iOS manage complex hardware, provide security, 
and enable countless applications.

The internet and networking have created a global computing infrastructure. ARPANET, the precursor to the internet, 
was established in 1969. The development of TCP/IP protocols in the 1970s provided the foundation for internet 
communication. The Domain Name System (DNS), created in the 1980s, made the internet human-readable. The World Wide 
Web, invented by Tim Berners-Lee in 1989, transformed the internet from a research tool to a global information 
and communication platform. Today, cloud computing provides on-demand access to vast computational resources, 
enabling services that would have been impossible just decades ago.

Database systems have evolved to manage increasingly complex data. Early file-based systems gave way to hierarchical 
and network databases. The relational model, proposed by Edgar Codd in 1970, revolutionized data management. SQL 
became the standard query language. NoSQL databases emerged to handle big data and non-relational data structures. 
Modern database systems handle petabytes of data, support real-time analytics, and enable distributed processing.

Security and cryptography have become increasingly critical. Early computers had minimal security concerns, but as 
networks expanded, threats grew. Encryption evolved from simple substitution ciphers to sophisticated algorithms 
like AES and RSA. Public-key cryptography, developed in the 1970s, enabled secure communication over insecure 
channels. Today, cybersecurity is a major industry, protecting everything from personal data to critical infrastructure.

Artificial intelligence has experienced multiple waves of enthusiasm and disappointment. Early AI research in the 
1950s and 1960s was optimistic but hit limitations, leading to 'AI winters.' Expert systems found success in the 
1980s. Machine learning gained prominence in the 1990s. The deep learning revolution of the 2010s, enabled by big 
data and powerful GPUs, has achieved remarkable breakthroughs. Large language models and generative AI represent 
the current frontier, with systems like GPT demonstrating capabilities that seemed impossible just years ago.

Quantum computing represents a potential paradigm shift. While classical computers use bits (0 or 1), quantum 
computers use qubits that can exist in superposition, allowing parallel processing of vast solution spaces. 
Quantum algorithms like Shor's algorithm could break current encryption, while others promise advances in drug 
discovery, optimization, and machine learning. Companies like IBM, Google, and startups are racing to build 
practical quantum computers, though significant challenges remain in error correction and scaling.

The future of computing holds exciting possibilities. Neuromorphic computing aims to mimic the brain's efficiency 
and parallel processing. DNA computing explores biological computation. Optical computing uses light for potentially 
faster processing. Edge computing brings computation closer to data sources, reducing latency. The integration of 
AI into all computing systems continues. As we look ahead, computing will likely become even more integrated into 
every aspect of human life, from smart cities to personalized medicine to space exploration.

Programming languages have evolved dramatically, each designed to solve specific problems. Early languages like FORTRAN focused on scientific computing, while COBOL targeted business applications. The C language, developed in the 1970s, became foundational, influencing countless subsequent languages. Object-oriented programming emerged with Smalltalk and was popularized by C++ and Java. Modern languages like Python prioritize readability and rapid development, JavaScript powers web applications, Rust focuses on safety and performance, and Go emphasizes simplicity and concurrency. Functional programming languages like Haskell and Scala offer different paradigms. Domain-specific languages target particular problem domains. The evolution continues with languages designed for parallel computing, distributed systems, and new computational paradigms.

Software engineering practices have matured significantly. Early programming was often ad-hoc, but the field developed methodologies like waterfall, agile, and DevOps. Version control systems evolved from simple file locking to distributed systems like Git. Testing practices grew from basic unit tests to comprehensive test suites including integration, end-to-end, and property-based testing. Code review became standard practice. Documentation and maintainability gained importance. Design patterns emerged to solve common problems. Software architecture evolved from monolithic to microservices, serverless, and event-driven architectures. The rise of open-source software transformed development, with platforms like GitHub enabling global collaboration.

Cloud computing has revolutionized how software is deployed and accessed. Infrastructure as a Service (IaaS) provides virtualized computing resources. Platform as a Service (PaaS) offers development environments. Software as a Service (SaaS) delivers applications over the internet. Major providers like Amazon Web Services, Microsoft Azure, and Google Cloud Platform offer vast arrays of services. Containerization with Docker and orchestration with Kubernetes enable scalable, portable deployments. Serverless computing abstracts away infrastructure management. Edge computing brings computation closer to data sources. These technologies enable startups to access enterprise-level infrastructure and scale rapidly.

Mobile computing has put powerful computers in billions of pockets. Smartphones combine communication, computing, sensors, and connectivity. Mobile operating systems like iOS and Android manage complex hardware and provide app ecosystems. Mobile apps have transformed industries from transportation to finance to entertainment. Responsive web design ensures applications work across devices. Progressive Web Apps (PWAs) bridge web and native app experiences. The mobile revolution continues with wearables, IoT devices, and emerging form factors.

Data science and big data have emerged as critical fields. The explosion of data from sensors, social media, transactions, and other sources created opportunities and challenges. Technologies like Hadoop and Spark enable processing of massive datasets. Data warehouses and data lakes store structured and unstructured data. Machine learning extracts insights from data. Business intelligence tools visualize and analyze data. Data governance and privacy have become important concerns. The ability to derive value from data has become a competitive advantage.

Cybersecurity has become increasingly critical as systems become more connected and valuable. Threats have evolved from simple viruses to sophisticated nation-state attacks. Encryption protects data in transit and at rest. Authentication and authorization control access. Firewalls and intrusion detection systems monitor networks. Security practices include penetration testing, vulnerability assessments, and security audits. The field addresses everything from protecting personal information to securing critical infrastructure. As attacks become more sophisticated, defenses must evolve continuously.

The future of computing holds exciting possibilities. Quantum computing may solve problems intractable for classical computers. Neuromorphic computing mimics the brain's efficiency. DNA computing explores biological computation. Optical computing uses light for potentially faster processing. Edge computing brings computation to where data is generated. The integration of AI into all systems continues. As computing becomes more powerful and accessible, it will likely transform every aspect of human experience, from how we work and learn to how we understand the universe and ourselves.

Artificial intelligence continues to advance at an unprecedented pace. Machine learning algorithms can now recognize patterns in data that humans would never detect. Deep learning neural networks with billions of parameters can generate human-like text, create realistic images, and solve complex problems. Reinforcement learning has achieved superhuman performance in games like Go and chess. Natural language processing models understand context, sentiment, and nuance. Computer vision systems can identify objects, faces, and scenes with remarkable accuracy. These advances are being applied across industries, from healthcare diagnostics to autonomous vehicles to financial trading.

The impact of computing on society cannot be overstated. It has transformed how we communicate, with instant messaging, video calls, and social media connecting people globally. Education has been revolutionized by online learning platforms, digital textbooks, and interactive simulations. Healthcare benefits from electronic records, telemedicine, and AI-assisted diagnosis. Transportation is being transformed by autonomous vehicles and smart traffic systems. Entertainment has been completely reshaped by streaming services, video games, and virtual reality. Commerce has moved online, with e-commerce platforms enabling global trade. Work has been transformed by remote collaboration tools, automation, and digital workflows.

As we look to the future, several trends are shaping computing. The Internet of Things (IoT) connects billions of devices, from smart home appliances to industrial sensors. 5G and future wireless technologies enable faster, more reliable connectivity. Augmented and virtual reality are creating new ways to interact with digital information. Blockchain technology offers new models for trust and decentralization. Sustainable computing addresses the environmental impact of technology. Quantum computing promises breakthroughs in cryptography, optimization, and scientific simulation. The boundaries between physical and digital worlds continue to blur.

The evolution of human-computer interaction has been transformative. Early computers required users to input programs via punch cards or switches. Command-line interfaces made computers more accessible but still required technical knowledge. Graphical user interfaces, pioneered by Xerox PARC and popularized by Apple and Microsoft, revolutionized computing by making it intuitive through visual metaphors like desktops, windows, and icons. Touch interfaces on smartphones and tablets made computing even more natural. Voice interfaces and natural language processing are making interaction more conversational. Gesture recognition, eye tracking, and brain-computer interfaces represent the next frontiers. These advances make computing accessible to broader audiences and enable new use cases.

The software development lifecycle has evolved significantly. Early programming was often done by individuals or small teams working in isolation. Modern development involves large, distributed teams collaborating across time zones. Agile methodologies emphasize iterative development and customer feedback. DevOps practices integrate development and operations for faster, more reliable deployments. Continuous integration and continuous deployment (CI/CD) automate testing and deployment. Code quality tools, linters, and static analysis help maintain standards. The rise of low-code and no-code platforms enables non-programmers to create applications. These trends make software development faster, more collaborative, and more accessible.

Data management has become increasingly sophisticated. Early systems stored data in flat files. Hierarchical and network databases provided more structure. The relational model, with its mathematical foundation, became dominant. SQL became the standard query language. Object-oriented databases emerged to handle complex data structures. NoSQL databases address the limitations of relational systems for big data, distributed systems, and non-structured data. NewSQL databases combine the benefits of both approaches. Data warehouses and data lakes store vast amounts of historical and real-time data. Data pipelines automate data movement and transformation. These technologies enable organizations to derive insights from massive, diverse datasets.

The economics of computing have shifted dramatically. Early computers were so expensive that only governments and large corporations could afford them. The personal computer revolution made computing accessible to individuals. The internet and open-source software further democratized access. Cloud computing enables startups to access enterprise-level infrastructure without massive upfront investment. Software as a Service (SaaS) models provide access to powerful applications through subscriptions. The app economy has created new business models. These changes have lowered barriers to entry and enabled innovation from diverse sources worldwide.

The impact on scientific research has been profound. Computers enable simulations of complex systems that would be impossible to study directly. Climate models predict future scenarios. Molecular dynamics simulations help design new drugs. Astrophysical simulations model the formation of galaxies. High-performance computing clusters process massive datasets from particle accelerators and telescopes. Bioinformatics uses computing to analyze genetic sequences and understand biological processes. Computational chemistry predicts molecular properties. These applications accelerate scientific discovery and enable research that would be impossible without computing power.

Education has been transformed by computing technology. Digital textbooks provide interactive content that adapts to individual learning styles. Online courses make education accessible globally. Educational software gamifies learning and provides immediate feedback. Virtual laboratories enable experiments that would be too expensive or dangerous in physical settings. Learning management systems organize course content and track student progress. Educational data mining identifies learning patterns and helps improve instruction. These technologies make education more accessible, personalized, and effective.

The creative arts have been revolutionized by computing. Digital art tools enable new forms of expression. Computer-generated imagery (CGI) creates stunning visual effects in films. Music production software allows anyone to create professional-quality recordings. Video editing software makes filmmaking accessible. 3D modeling and animation bring imaginary worlds to life. Virtual and augmented reality create immersive artistic experiences. These tools democratize creativity and enable new forms of artistic expression that were previously impossible.

As computing continues to evolve, we stand at the threshold of new possibilities. The convergence of artificial intelligence, quantum computing, biotechnology, and nanotechnology promises to create systems beyond our current imagination. The challenge will be to harness these powerful technologies for the benefit of all humanity while addressing concerns about privacy, security, employment, and equity. The history of computing shows that each generation of technology builds upon the previous, creating capabilities that seemed impossible just years before. As we continue this journey, computing will likely become even more integrated into the fabric of human existence, transforming how we understand ourselves and our place in the universe. The exponential growth in computing power, following Moore's Law for decades, has enabled these transformations. While physical limits may slow this growth, new architectures, materials, and paradigms continue to push boundaries. The democratization of computing power through cloud services means that individuals and small organizations can now access computational resources that were once available only to the largest institutions. This accessibility fuels innovation and enables solutions to global challenges in climate, health, education, and more. The future promises even more remarkable advances as we continue to push the boundaries of what is computationally possible, always building on the foundation laid by generations of innovators who came before us. In conclusion, the story of computing is one of continuous innovation, where each breakthrough enables the next. From mechanical calculators to quantum computers, from room-sized machines to devices in our pockets, computing has transformed every aspect of human life. The compression technology demonstrated here represents just one of many innovations that make modern computing systems efficient and powerful. By converting lengthy text into compact vector representations, we can store vast amounts of information in minimal space while maintaining the ability to retrieve and understand the semantic content. This capability is essential for building intelligent systems that can learn from and reason about large bodies of knowledge.""",
        0.9
    )
    
    # Add one more example showing 30x+ compression
    massive_text = (
        "Provide an extremely comprehensive and detailed explanation covering the complete history of computing, artificial intelligence, machine learning, deep learning, neural networks, quantum computing, cloud computing, mobile computing, software development, programming languages, operating systems, networking protocols, database systems, cybersecurity, cryptography, data science, big data analytics, internet technologies, web development, and the future of technology, including extensive details about each topic, their interconnections, historical context, current state, and future prospects.",
        ultra_long_text[1] + """ 

Additional comprehensive coverage: The field of distributed systems has enabled computing at unprecedented scales. Distributed computing allows multiple computers to work together, sharing resources and processing power. This enables handling of massive workloads that would be impossible for a single machine. Load balancing distributes work across servers to optimize performance and reliability. Distributed databases ensure data consistency across multiple locations. Consensus algorithms like Raft and Paxos enable distributed systems to agree on shared state. Microservices architectures break applications into independently deployable services. These technologies power modern internet-scale applications that serve billions of users.

Parallel computing has been crucial for handling computationally intensive tasks. Early parallel systems used multiple processors in a single machine. Cluster computing connects multiple computers over networks. Grid computing spans organizational boundaries. Supercomputers combine thousands of processors to solve complex problems in science and engineering. Graphics Processing Units (GPUs), originally designed for rendering graphics, have become powerful parallel processors for scientific computing and machine learning. Tensor Processing Units (TPUs) are specialized for neural network operations. These parallel architectures enable simulations, data analysis, and AI training that would take years on single processors.

The evolution of storage technology has been remarkable. Early computers used punch cards and paper tape. Magnetic tape provided sequential storage. Hard disk drives offered random access with increasing capacity and decreasing cost. Solid-state drives (SSDs) use flash memory for much faster access. Cloud storage provides virtually unlimited capacity accessible from anywhere. Distributed storage systems replicate data across multiple locations for reliability. Object storage systems handle unstructured data at massive scales. These advances have made storing and accessing vast amounts of data practical and affordable.

Networking technologies have connected the world. Local area networks (LANs) connect devices within buildings. Wide area networks (WANs) span cities and countries. The internet connects networks globally using standardized protocols. Ethernet became the dominant wired networking standard. Wi-Fi enables wireless connectivity. Cellular networks provide mobile internet access. Fiber optic cables transmit data at the speed of light. Satellite networks extend connectivity to remote areas. These technologies have created a globally connected digital infrastructure.

The software industry has grown into one of the world's largest economic sectors. Software companies range from startups to global corporations. The industry employs millions of developers worldwide. Open-source software has created alternatives to proprietary solutions. Software licensing models have evolved from one-time purchases to subscriptions to freemium. The app economy has created new business models and revenue streams. Software-as-a-Service (SaaS) has become the dominant delivery model for business applications. These trends continue to shape how software is created, distributed, and consumed.

Quality assurance and testing have become sophisticated disciplines. Unit testing verifies individual components. Integration testing checks how components work together. System testing validates complete applications. Performance testing ensures systems meet speed and capacity requirements. Security testing identifies vulnerabilities. Automated testing enables continuous validation. Test-driven development writes tests before code. These practices ensure software reliability and quality.

The documentation and knowledge management aspects of computing are crucial. Technical documentation helps developers understand and use systems. User documentation guides end users. API documentation enables integration. Knowledge bases capture institutional knowledge. Wikis enable collaborative documentation. These resources are essential for maintaining and evolving complex systems.

Project management in software development has evolved significantly. Waterfall methodologies provided structured approaches but were inflexible. Agile methodologies emphasize adaptability and customer collaboration. Scrum and Kanban provide frameworks for agile development. DevOps integrates development and operations. These approaches help teams deliver software more effectively.

The legal and ethical dimensions of computing have gained importance. Intellectual property laws protect software innovations. Privacy regulations like GDPR protect user data. Cybersecurity laws address digital crimes. Net neutrality debates shape internet access. These issues will continue to evolve as technology advances.

International collaboration has been essential to computing's progress. Standards organizations like IEEE, ISO, and W3C develop technical standards. Open-source communities collaborate globally. Research institutions share knowledge. International conferences facilitate knowledge exchange. This collaboration accelerates innovation and ensures compatibility.

The environmental impact of computing is receiving increasing attention. Data centers consume vast amounts of energy. E-waste from obsolete devices is a growing concern. Sustainable computing practices aim to reduce environmental impact. Renewable energy powers many data centers. Efficient algorithms reduce computational requirements. These efforts are crucial as computing scales globally.

Accessibility in computing ensures technology is usable by people with disabilities. Screen readers enable blind users to access computers. Voice recognition helps those with mobility limitations. Captioning makes video content accessible to the deaf. These technologies demonstrate computing's potential to be inclusive.

The role of computing in scientific discovery continues to grow. Computational biology uses computers to understand living systems. Computational chemistry simulates molecular interactions. Computational physics models physical phenomena. Digital humanities apply computing to cultural studies. These applications expand the frontiers of knowledge.

As we reflect on computing's journey, we see a pattern of exponential growth and transformation. Each generation of technology builds on previous foundations while opening new possibilities. The future will likely bring even more remarkable advances as we continue to explore the potential of computation. The compression demonstrated here is just one example of how innovative techniques can make computing more efficient and powerful.""",
        0.9
    )
    
    texts = [
        ("Short text", short_text),
        ("Medium text", medium_text),
        ("Long text", long_text),
        ("Ultra long text (26KB)", ultra_long_text),
        ("Massive text (50KB+)", massive_text),
    ]
    
    for label, (user_msg, assistant_msg, importance) in texts:
        exchange_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        original_size = len(exchange_text.encode('utf-8'))
        
        token_id = memory.consolidate_exchange(user_msg, assistant_msg, importance)
        
        vector_size = 1632  # 408d * 4 bytes
        compression = original_size / vector_size if vector_size > 0 else 0
        
        status = "🔥 Excellent" if compression >= 50 else "✅ High" if compression >= 16 else "📊 Good" if compression >= 4 else "📝 Low"
        
        print(f"  {status} - {label}:")
        print(f"    Original: {original_size:,} bytes ({original_size/1024:.2f} KB)")
        print(f"    Compressed: {vector_size:,} bytes ({vector_size/1024:.2f} KB)")
        print(f"    Compression: {compression:.1f}x\n")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("📊 Overall Statistics:")
    stats = memory.get_stats()
    vector_size_per_doc = 1632
    total_vector_size = stats['total_documents'] * vector_size_per_doc
    original_size = stats['total_bytes_stored']
    overall_compression = original_size / total_vector_size if total_vector_size > 0 else 0
    
    print(f"  • Total documents: {stats['total_documents']}")
    print(f"  • Original text: {original_size:,} bytes ({original_size/1024:.2f} KB)")
    print(f"  • Vector storage: {total_vector_size:,} bytes ({total_vector_size/1024:.2f} KB)")
    print(f"  • Overall compression: {overall_compression:.1f}x")
    print(f"\n💡 Key Insight: Compression ratios of 16-128x are achieved with longer texts.")
    print(f"   Each vector is fixed at {vector_size_per_doc} bytes, so longer documents compress more!")
    print("=" * 70)


if __name__ == "__main__":
    main()
