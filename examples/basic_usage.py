"""
KAIROS Memory - Basic Usage Example

Demonstrates how to use KAIROS in any GenAI application.
"""
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kairos import KAIROSMemory


def main():
    print("=" * 60)
    print("KAIROS Memory - Basic Usage Example")
    print("=" * 60)
    
    # Initialize memory system
    memory = KAIROSMemory(
        storage_path="./example_memory",
        use_multidim=True,
        enable_feedback=True
    )
    
    # Store conversation exchanges with LONG text to demonstrate compression
    # Compression benefits (16-128x) only appear with longer texts since vectors are fixed-size
    print("\n📝 Storing conversation exchanges (demonstrating compression with longer texts)...")
    
    exchanges = [
        (
            "Can you provide a comprehensive explanation of how deep learning neural networks work, including their architecture, training process, and real-world applications?", 
            """Deep learning neural networks are sophisticated machine learning models inspired by the structure and function of the human brain. At their core, they consist of multiple layers of interconnected nodes (neurons) that process information in a hierarchical manner. The architecture typically includes an input layer that receives raw data, one or more hidden layers that perform complex transformations, and an output layer that produces predictions or classifications.

The training process involves feeding large amounts of labeled data through the network, calculating the error between predictions and actual values, and then using backpropagation to adjust the weights and biases of connections throughout the network. This iterative process continues until the network learns to make accurate predictions. The key innovation in deep learning is the ability to automatically learn hierarchical representations - lower layers detect simple features like edges, while higher layers combine these into more complex patterns.

Real-world applications are vast and growing. In computer vision, deep learning powers facial recognition systems, autonomous vehicles that can detect pedestrians and obstacles, and medical imaging systems that can identify diseases from X-rays and MRIs. In natural language processing, these networks enable machine translation, chatbots, sentiment analysis, and text generation. Speech recognition systems like Siri and Alexa rely heavily on deep learning. The technology also revolutionizes fields like drug discovery, where it can predict molecular properties, and finance, where it detects fraudulent transactions. Recommendation systems used by Netflix and Amazon use deep learning to understand user preferences. The field continues to evolve rapidly, with new architectures and techniques emerging regularly.""",
            0.9
        ),
        (
            "I'd like to understand the complete history and evolution of artificial intelligence from its origins to the present day.",
            """The history of artificial intelligence is a fascinating journey spanning over 70 years. The field officially began in 1956 at the Dartmouth Conference, where the term 'artificial intelligence' was coined by John McCarthy. Early AI research in the 1950s and 1960s was characterized by great optimism - researchers believed machines would achieve human-level intelligence within decades. This period saw the development of early programs like the Logic Theorist and the General Problem Solver, which could solve mathematical theorems and puzzles.

The 1970s and 1980s brought the first 'AI winter' - a period of reduced funding and interest due to unmet expectations. However, this era also saw important developments in expert systems, which encoded human knowledge into rule-based programs. These systems found success in specialized domains like medical diagnosis and financial analysis. The 1990s marked a shift toward machine learning approaches, with statistical methods gaining prominence. IBM's Deep Blue defeating world chess champion Garry Kasparov in 1997 was a major milestone.

The 2000s saw the rise of big data and improved computational power, enabling more sophisticated machine learning algorithms. The 2010s marked the deep learning revolution, driven by advances in neural network architectures, better training algorithms, and the availability of massive datasets. Breakthroughs like AlexNet in 2012, which dramatically improved image recognition, and AlphaGo defeating the world Go champion in 2016, demonstrated the power of modern AI.

Today, we're in an era of large language models and generative AI. Systems like GPT, BERT, and their successors have transformed natural language processing. AI is now integrated into countless applications - from search engines and social media to healthcare, transportation, and scientific research. The field continues to evolve rapidly, with ongoing debates about AI safety, ethics, and the potential for artificial general intelligence.""",
            0.9
        ),
        (
            "Please explain the detailed process of how the human brain processes and stores memories, including the different types of memory and the neural mechanisms involved.",
            """Memory formation and storage in the human brain is an extraordinarily complex process involving multiple brain regions and neural mechanisms. The process begins with encoding, where sensory information is transformed into a neural code that the brain can store. This happens primarily in the hippocampus, a seahorse-shaped structure deep in the temporal lobe, which acts as a sort of 'memory gateway.'

There are several distinct types of memory. Sensory memory holds raw sensory information for just milliseconds to seconds - like the brief afterimage you see after a camera flash. Short-term or working memory can hold information for seconds to minutes, with a limited capacity of about 7±2 items. This type of memory relies heavily on the prefrontal cortex and involves active rehearsal to maintain information. Long-term memory, which can last from hours to a lifetime, is divided into explicit (declarative) and implicit (non-declarative) memories.

Explicit memories include episodic memories (personal experiences and events) and semantic memories (facts and general knowledge). These are initially processed in the hippocampus but gradually become consolidated in the neocortex, particularly in regions related to the original sensory experience. Implicit memories include procedural memories (skills and habits), which involve the basal ganglia and cerebellum, and emotional memories, which involve the amygdala.

The neural mechanism of memory storage involves changes in synaptic strength - the connections between neurons. This process, called long-term potentiation (LTP), strengthens synapses that are repeatedly activated together, following the principle 'neurons that fire together, wire together.' Protein synthesis is crucial for long-term memory formation, creating new proteins that modify synapses and even grow new connections. Sleep plays a vital role in memory consolidation, with different sleep stages helping to transfer memories from the hippocampus to long-term storage in the cortex.

Memory retrieval involves reactivating the same neural patterns that were active during encoding, a process that can actually modify and strengthen the original memory through reconsolidation. The brain's memory system is remarkably distributed and redundant, which is why damage to one area rarely causes complete memory loss.""",
            0.9
        ),
        (
            "I want a comprehensive guide on how to build a successful software development career, including education paths, essential skills, career progression, and industry best practices.",
            """Building a successful software development career requires a combination of technical skills, continuous learning, practical experience, and professional development. The education path is more flexible than ever - while computer science degrees from universities remain valuable, many successful developers are self-taught or come from coding bootcamps. The key is demonstrating competency through projects and practical skills rather than just credentials.

Essential technical skills start with proficiency in at least one programming language - Python, JavaScript, Java, or C++ are excellent starting points. Understanding data structures and algorithms is crucial for problem-solving. Version control systems like Git are essential tools. You should be comfortable with databases (both SQL and NoSQL), understand web development fundamentals (HTML, CSS, JavaScript), and have experience with frameworks relevant to your chosen path. Cloud computing knowledge (AWS, Azure, or GCP) is increasingly important. Understanding software development methodologies like Agile and DevOps practices is also valuable.

Career progression typically follows a path from junior developer to mid-level, then senior developer, and potentially to lead developer, architect, or engineering manager roles. Each level requires increasing responsibility, technical depth, and often leadership skills. Building a portfolio of projects is crucial - contribute to open source, build personal projects, and document your work on platforms like GitHub. Networking through meetups, conferences, and online communities can open opportunities.

Industry best practices include writing clean, maintainable code with proper documentation. Understanding testing (unit, integration, end-to-end) is essential. Code reviews are standard practice and a great learning opportunity. Stay current with industry trends but don't chase every new technology - depth in core technologies is often more valuable than breadth. Develop soft skills like communication, teamwork, and problem-solving. Consider specializing in areas like mobile development, data science, cybersecurity, or cloud architecture as your career progresses. Remember that software development is a field of continuous learning - technologies evolve rapidly, and successful developers embrace lifelong learning.""",
            0.9
        ),
        (
            "Can you provide a detailed explanation of climate change, including its causes, effects, scientific evidence, and potential solutions?",
            """Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities since the mid-20th century. The scientific consensus is overwhelming - Earth's climate is warming, and human activities are the primary driver. The main cause is the enhanced greenhouse effect, where certain gases in the atmosphere trap heat that would otherwise escape into space.

The primary greenhouse gas is carbon dioxide (CO2), released through burning fossil fuels (coal, oil, natural gas) for energy, transportation, and industrial processes. Deforestation also contributes significantly, as trees absorb CO2. Other important greenhouse gases include methane (from agriculture, landfills, and fossil fuel extraction), nitrous oxide (from agriculture and industrial processes), and fluorinated gases (from industrial applications). Since the Industrial Revolution, atmospheric CO2 levels have increased from about 280 parts per million to over 420 ppm today.

The effects of climate change are already visible and accelerating. Global average temperatures have risen approximately 1.1°C since pre-industrial times. This warming causes sea levels to rise through thermal expansion and melting ice sheets and glaciers. Extreme weather events are becoming more frequent and intense - heatwaves, droughts, floods, and hurricanes. Ocean acidification threatens marine ecosystems. Changes in precipitation patterns affect agriculture and water supplies. Many species face extinction as their habitats change faster than they can adapt.

Scientific evidence comes from multiple sources: direct temperature measurements show consistent warming trends, ice core samples reveal historical climate data, satellite observations track changes in ice sheets and sea levels, and climate models successfully predict observed changes. The Intergovernmental Panel on Climate Change (IPCC) synthesizes thousands of scientific studies and provides authoritative assessments.

Potential solutions require action at multiple levels. Transitioning to renewable energy sources (solar, wind, hydroelectric) is crucial. Improving energy efficiency in buildings, transportation, and industry can significantly reduce emissions. Reforestation and protecting existing forests helps absorb CO2. Developing carbon capture and storage technologies may be necessary. Individual actions matter, but systemic change through government policies, corporate responsibility, and international cooperation is essential. The Paris Agreement represents a global commitment to limit warming, though current pledges need strengthening to meet ambitious targets.""",
            0.9
        ),
        (
            "Explain the complete process of how modern web browsers work, from receiving a URL to displaying a webpage, including all the technical steps involved.",
            """When you type a URL into a browser and press Enter, a complex sequence of events unfolds behind the scenes. The process begins with URL parsing - the browser breaks down the URL into its components: protocol (http/https), domain name, path, and optional parameters. The browser checks its cache first to see if the page was recently visited, which can dramatically speed up loading.

If not cached, DNS (Domain Name System) resolution occurs. The browser queries DNS servers to translate the human-readable domain name (like 'example.com') into an IP address (like 192.0.2.1). This involves checking the browser's DNS cache, then the operating system's cache, then querying DNS servers in a hierarchical process. Once the IP address is obtained, the browser can establish a connection.

The browser initiates a TCP (Transmission Control Protocol) connection to the server's IP address on the appropriate port (80 for HTTP, 443 for HTTPS). For HTTPS, an additional TLS (Transport Layer Security) handshake occurs to establish an encrypted connection. This involves certificate verification, key exchange, and establishing encryption parameters. This secure connection ensures data privacy and integrity.

Once connected, the browser sends an HTTP request. This includes the request method (GET, POST, etc.), headers (browser type, accepted content types, cookies), and the requested path. The server processes this request, potentially querying databases, executing server-side code, and assembling the response. The server sends back an HTTP response containing status codes, headers, and the actual content (HTML, CSS, JavaScript, images, etc.).

The browser receives the response and begins parsing. The HTML parser builds a Document Object Model (DOM) tree, representing the page structure. As it encounters external resources (CSS files, JavaScript files, images), it makes additional requests. CSS is parsed to build the CSS Object Model (CSSOM), which determines styling rules. JavaScript can modify the DOM and CSSOM, so parsing may pause for script execution.

The browser combines the DOM and CSSOM to create a render tree, which includes only visible elements with their computed styles. Layout (or reflow) calculates the exact position and size of each element. Then painting occurs, where the browser fills in pixels for each element. Modern browsers use compositing to optimize rendering, separating layers that can be updated independently.

The browser also handles JavaScript execution through its JavaScript engine (like V8 in Chrome). This includes parsing, compilation, and execution. Modern engines use just-in-time (JIT) compilation for performance. The browser manages the event loop, handling asynchronous operations, timers, and user interactions. Throughout this process, the browser optimizes performance through techniques like lazy loading, prefetching, and caching strategies.""",
            0.9
        ),
    ]
    
    for user_msg, assistant_msg, importance in exchanges:
        token_id = memory.consolidate_exchange(user_msg, assistant_msg, importance)
        print(f"  ✓ Stored: '{user_msg[:40]}...' (ID: {token_id})")
    
    # Retrieve relevant memories
    print("\n🔍 Retrieving relevant memories...")
    
    queries = [
        "What food do I like?",
        "Tell me about AI and machine learning",
        "What's my favorite color?",
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results, session_id = memory.retrieve_relevant(query, top_k=2)
        
        if results:
            for i, mem in enumerate(results, 1):
                print(f"    {i}. (sim={mem['similarity']:.2f}) {mem['content'][:60]}...")
        else:
            print("    No results found")
    
    # Show statistics
    print("\n📊 Memory Statistics:")
    stats = memory.get_stats()
    print(f"  • Total documents: {stats['total_documents']}")
    print(f"  • Total stores: {stats['stores']}")
    print(f"  • Total retrievals: {stats['retrievals']}")
    print(f"  • Avg store latency: {stats['store_latency_ms']['mean']:.1f}ms")
    print(f"  • Avg retrieve latency: {stats['retrieve_latency_ms']['mean']:.1f}ms")
    print(f"  • Storage size: {stats['storage_size_mb']:.3f}MB")
    
    # Compression statistics with detailed breakdown
    if stats['total_bytes_stored'] > 0 and stats['total_documents'] > 0:
        vector_size_per_doc = 1632 if stats['multidim_enabled'] else 1536
        estimated_vector_size = stats['total_documents'] * vector_size_per_doc
        original_size = stats['total_bytes_stored']
        compression_ratio = original_size / estimated_vector_size if estimated_vector_size > 0 else 0
        avg_text_per_doc = original_size / stats['total_documents']
        
        print(f"\n💾 Compression Statistics:")
        print(f"  • Original text: {original_size:,} bytes ({original_size/1024:.2f} KB)")
        print(f"  • Average text per document: {avg_text_per_doc:.0f} bytes")
        print(f"  • Vector storage: ~{estimated_vector_size:,} bytes ({estimated_vector_size/1024:.2f} KB)")
        print(f"  • Vector size per document: {vector_size_per_doc} bytes (fixed)")
        print(f"  • Overall compression ratio: {compression_ratio:.2f}x")
        
        # Show per-document compression to highlight high compression cases
        print(f"\n  Per-document compression examples:")
        all_tokens = memory.latent_store.get_all_tokens()
        compression_examples = []
        for token_id, data in list(all_tokens.items())[:3]:  # Show first 3 as examples
            meta = data.get('metadata', {})
            # Estimate original text size from metadata
            exchange_length = meta.get('exchange_length', 0)
            if exchange_length > 0:
                doc_compression = exchange_length / vector_size_per_doc
                compression_examples.append((exchange_length, doc_compression))
        
        if compression_examples:
            for orig_size, comp_ratio in compression_examples:
                status = "✅ High compression" if comp_ratio >= 16 else "📊 Moderate" if comp_ratio >= 4 else "📝 Short text"
                print(f"    - {orig_size:,} bytes → {vector_size_per_doc} bytes = {comp_ratio:.1f}x {status}")
        
        if compression_ratio < 16:
            print(f"\n  ℹ️  Note: Compression ratios of 16-128x are achieved with longer texts (>26KB per document).")
            print(f"      Current average text size ({avg_text_per_doc:.0f} bytes) shows compression benefits increase with text length.")
            print(f"      Run 'python examples/compression_demo.py' to see examples achieving 16x+ compression!")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
