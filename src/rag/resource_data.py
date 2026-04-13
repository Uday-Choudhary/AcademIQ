"""
Curated educational resource data for the RAG knowledge base.
Each resource is categorized by subject, level, and includes a description for embedding.
"""

STUDY_RESOURCES = [
    # ─── MATH ────────────────────────────────────────────────────────────
    {
        "title": "Khan Academy — Algebra",
        "url": "https://www.khanacademy.org/math/algebra",
        "subject": "Math",
        "level": "Beginner",
        "description": "Free comprehensive algebra course with practice problems. Covers linear equations, inequalities, functions, and graphing. Interactive exercises with instant feedback. Great for students scoring below 50 in math."
    },
    {
        "title": "Khan Academy — Geometry",
        "url": "https://www.khanacademy.org/math/geometry",
        "subject": "Math",
        "level": "Beginner",
        "description": "Visual geometry lessons covering angles, triangles, circles, area, and proofs. Interactive exercises with step-by-step hints. Ideal for building spatial reasoning skills."
    },
    {
        "title": "Khan Academy — Calculus",
        "url": "https://www.khanacademy.org/math/calculus-1",
        "subject": "Math",
        "level": "Advanced",
        "description": "Calculus fundamentals: limits, derivatives, integrals. Video lectures with practice problems. For advanced students looking to strengthen analytical skills."
    },
    {
        "title": "PatrickJMT — Math Video Tutorials",
        "url": "https://www.patrickjmt.com",
        "subject": "Math",
        "level": "Intermediate",
        "description": "Short, focused math tutorial videos covering algebra, calculus, and statistics. Excellent for reviewing specific topics and exam preparation."
    },
    {
        "title": "MIT OpenCourseWare — Mathematics",
        "url": "https://ocw.mit.edu/courses/mathematics/",
        "subject": "Math",
        "level": "Advanced",
        "description": "University-level mathematics courses from MIT. Includes linear algebra, differential equations, and probability. Lecture notes, problem sets, and exams available for free."
    },
    {
        "title": "Mathway — Step-by-Step Problem Solver",
        "url": "https://www.mathway.com",
        "subject": "Math",
        "level": "Beginner",
        "description": "Math problem solver that shows step-by-step solutions. Covers basic math, algebra, trigonometry, and calculus. Useful for checking homework and understanding solution methods."
    },
    {
        "title": "Brilliant.org — Math Fundamentals",
        "url": "https://brilliant.org/courses/math-fundamentals/",
        "subject": "Math",
        "level": "Intermediate",
        "description": "Interactive math courses that build problem-solving skills through guided challenges. Focuses on conceptual understanding rather than rote memorization."
    },
    {
        "title": "3Blue1Brown — Essence of Linear Algebra",
        "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab",
        "subject": "Math",
        "level": "Advanced",
        "description": "Beautiful visual explanations of linear algebra concepts. Helps develop deep intuition for vectors, matrices, and transformations. Perfect for visual learners."
    },

    # ─── SCIENCE ─────────────────────────────────────────────────────────
    {
        "title": "Khan Academy — Biology",
        "url": "https://www.khanacademy.org/science/biology",
        "subject": "Science",
        "level": "Beginner",
        "description": "Comprehensive biology course covering cells, genetics, evolution, and ecology. Video lectures with practice questions. Great foundation for struggling science students."
    },
    {
        "title": "Khan Academy — Chemistry",
        "url": "https://www.khanacademy.org/science/chemistry",
        "subject": "Science",
        "level": "Beginner",
        "description": "Chemistry fundamentals: atoms, chemical bonds, reactions, and stoichiometry. Animated explanations with exercises. Ideal for students who find chemistry difficult."
    },
    {
        "title": "Khan Academy — Physics",
        "url": "https://www.khanacademy.org/science/physics",
        "subject": "Science",
        "level": "Intermediate",
        "description": "Physics concepts from mechanics to electricity. Clear explanations with real-world examples and practice problems. Perfect for building problem-solving intuition."
    },
    {
        "title": "CrashCourse — Science (YouTube)",
        "url": "https://www.youtube.com/c/crashcourse",
        "subject": "Science",
        "level": "Beginner",
        "description": "Engaging, fast-paced science videos covering biology, chemistry, physics, and more. Great for quick overviews and making science interesting and accessible."
    },
    {
        "title": "MIT OpenCourseWare — Physics",
        "url": "https://ocw.mit.edu/courses/physics/",
        "subject": "Science",
        "level": "Advanced",
        "description": "University-level physics courses from MIT. Includes classical mechanics, electromagnetism, and quantum physics. Full lecture videos, notes, and problem sets."
    },
    {
        "title": "PhET Interactive Simulations",
        "url": "https://phet.colorado.edu",
        "subject": "Science",
        "level": "Beginner",
        "description": "Free interactive science simulations from University of Colorado. Covers physics, chemistry, math, and biology. Hands-on virtual experiments for visual learning."
    },
    {
        "title": "Coursera — Science Courses",
        "url": "https://www.coursera.org/browse/physical-science-and-engineering",
        "subject": "Science",
        "level": "Intermediate",
        "description": "University-level science courses from top institutions. Structured learning with quizzes, assignments, and certificates. Audit mode available for free."
    },
    {
        "title": "Organic Chemistry Tutor (YouTube)",
        "url": "https://www.youtube.com/c/TheOrganicChemistryTutor",
        "subject": "Science",
        "level": "Intermediate",
        "description": "Detailed tutorial videos for chemistry, physics, and math. Step-by-step problem solving with clear explanations. Excellent exam preparation resource."
    },

    # ─── ENGLISH ─────────────────────────────────────────────────────────
    {
        "title": "Grammarly Blog — Grammar Guides",
        "url": "https://www.grammarly.com/blog/category/handbook/",
        "subject": "English",
        "level": "Beginner",
        "description": "Free grammar guides covering punctuation, sentence structure, and common errors. Clear explanations with examples. Essential for improving writing accuracy."
    },
    {
        "title": "Khan Academy — Grammar",
        "url": "https://www.khanacademy.org/humanities/grammar",
        "subject": "English",
        "level": "Beginner",
        "description": "Complete English grammar course covering parts of speech, syntax, and punctuation. Practice exercises with instant feedback. Perfect for students below 50 in English."
    },
    {
        "title": "Purdue OWL — Writing Resources",
        "url": "https://owl.purdue.edu/owl/purdue_owl.html",
        "subject": "English",
        "level": "Intermediate",
        "description": "Comprehensive writing resource covering essay writing, citation formats, grammar, and research papers. Industry-standard reference for academic writing."
    },
    {
        "title": "ReadTheory — Reading Comprehension",
        "url": "https://readtheory.org",
        "subject": "English",
        "level": "Beginner",
        "description": "Adaptive reading comprehension platform that adjusts difficulty based on performance. Free quizzes and passages. Improves reading speed and understanding."
    },
    {
        "title": "Coursera — English Composition",
        "url": "https://www.coursera.org/courses?query=english%20composition",
        "subject": "English",
        "level": "Intermediate",
        "description": "University-level English writing courses from top institutions. Covers essay structure, argumentation, and academic writing. Peer-reviewed assignments."
    },
    {
        "title": "Vocabulary.com",
        "url": "https://www.vocabulary.com",
        "subject": "English",
        "level": "Beginner",
        "description": "Adaptive vocabulary building platform with spaced repetition. Game-like interface makes learning new words engaging. Tracks progress over time."
    },
    {
        "title": "BBC Learning English",
        "url": "https://www.bbc.co.uk/learningenglish",
        "subject": "English",
        "level": "Beginner",
        "description": "Free English learning resources from BBC. Video lessons, grammar guides, pronunciation practice, and quizzes. Great for non-native English speakers."
    },
    {
        "title": "EdX — English Language & Literature",
        "url": "https://www.edx.org/learn/english",
        "subject": "English",
        "level": "Advanced",
        "description": "University-level English courses from Harvard, MIT, and more. Covers literature analysis, critical thinking, and advanced writing. Free audit available."
    },

    # ─── GENERAL STUDY SKILLS ────────────────────────────────────────────
    {
        "title": "Coursera — Learning How to Learn",
        "url": "https://www.coursera.org/learn/learning-how-to-learn",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "World's most popular online course on learning techniques. Covers spaced repetition, chunking, memory tricks, and overcoming procrastination. Essential for students with low study hours."
    },
    {
        "title": "Anki — Spaced Repetition Flashcards",
        "url": "https://apps.ankiweb.net",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Free flashcard app using spaced repetition algorithm. Create custom flashcards for any subject. Proven to improve long-term memory retention significantly."
    },
    {
        "title": "Pomodoro Technique Guide",
        "url": "https://todoist.com/productivity-methods/pomodoro-technique",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Time management method: study in 25-minute focused sessions with 5-minute breaks. Reduces procrastination and improves focus. Ideal for students struggling with study hours."
    },
    {
        "title": "Forest App — Focus Timer",
        "url": "https://www.forestapp.cc",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Gamified focus timer that grows virtual trees while you study. Helps build consistent study habits. Great for students who need to increase study hours."
    },
    {
        "title": "Notion — Student Template",
        "url": "https://www.notion.so/templates/category/school",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Free organizational templates for students. Plan study schedules, track assignments, and organize notes. Helps build discipline and time management skills."
    },
    {
        "title": "Thomas Frank — Study Tips (YouTube)",
        "url": "https://www.youtube.com/c/Thomasfrank",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Practical study tips, productivity hacks, and motivation videos. Evidence-based learning strategies explained in an engaging way. Great for improving study habits."
    },
    {
        "title": "Cal Newport — Deep Work Strategies",
        "url": "https://www.calnewport.com/blog/",
        "subject": "Study Skills",
        "level": "Intermediate",
        "description": "Strategies for deep, focused work and study. Covers techniques to eliminate distractions and maximize learning in limited time. Essential for serious academic improvement."
    },

    # ─── ATTENDANCE & MOTIVATION ─────────────────────────────────────────
    {
        "title": "Headspace — Student Mindfulness",
        "url": "https://www.headspace.com/students",
        "subject": "Wellness",
        "level": "Beginner",
        "description": "Free mindfulness and meditation app for students. Reduces stress and anxiety that may be causing low attendance. Guided sessions as short as 3 minutes."
    },
    {
        "title": "TED-Ed — Educational Videos",
        "url": "https://ed.ted.com",
        "subject": "General",
        "level": "Beginner",
        "description": "Short animated educational videos on science, math, and humanities. Makes learning fun and engaging. Great for reigniting interest in subjects where motivation is low."
    },
    {
        "title": "Study With Me Videos (YouTube)",
        "url": "https://www.youtube.com/results?search_query=study+with+me",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Virtual study companion videos. Study alongside others in real-time or pre-recorded sessions. Helps with accountability and consistent study habits."
    },

    # ─── TEST PREPARATION ────────────────────────────────────────────────
    {
        "title": "Quizlet — Study Flashcards",
        "url": "https://quizlet.com",
        "subject": "General",
        "level": "Beginner",
        "description": "Create and study flashcards for any subject. Community-made flashcard sets available. Multiple study modes including games and practice tests."
    },
    {
        "title": "Khan Academy — Test Prep",
        "url": "https://www.khanacademy.org/test-prep",
        "subject": "General",
        "level": "Intermediate",
        "description": "Free test preparation resources for standardized exams. Full-length practice tests, video explanations, and personalized practice recommendations."
    },
    {
        "title": "GCFGlobal — Free Learning",
        "url": "https://edu.gcfglobal.org",
        "subject": "General",
        "level": "Beginner",
        "description": "Free courses on core academics, technology, and life skills. Ideal for students without internet access at home who can access from school or library computers."
    },

    # ─── NO INTERNET ACCESS RESOURCES ────────────────────────────────────
    {
        "title": "OpenStax — Free Textbooks",
        "url": "https://openstax.org",
        "subject": "General",
        "level": "Intermediate",
        "description": "Free peer-reviewed textbooks in PDF format for math, science, and more. Can be downloaded once and used offline. Perfect for students without consistent internet access."
    },
    {
        "title": "Libby/OverDrive — Library eBooks",
        "url": "https://www.overdrive.com",
        "subject": "General",
        "level": "Beginner",
        "description": "Borrow free eBooks and audiobooks from your local library. Works offline after download. Access educational materials without internet through library membership."
    },
    {
        "title": "Kiwix — Offline Wikipedia & Khan Academy",
        "url": "https://www.kiwix.org",
        "subject": "General",
        "level": "Beginner",
        "description": "Download entire websites (Wikipedia, Khan Academy) for offline use. One-time download, then access educational content without internet. Essential for students with no internet at home."
    },

    # ─── PEER LEARNING & COLLABORATION ───────────────────────────────────
    {
        "title": "Chegg Study — Peer Tutoring",
        "url": "https://www.chegg.com/study",
        "subject": "General",
        "level": "Intermediate",
        "description": "Step-by-step textbook solutions and expert Q&A. Connects students with peer tutors. Useful for students who learn better with guided help and explanations."
    },
    {
        "title": "Discord Study Servers",
        "url": "https://discord.com",
        "subject": "Study Skills",
        "level": "Beginner",
        "description": "Join free study group servers on Discord. Real-time collaboration, Q&A channels by subject, and virtual study rooms. Great for students who benefit from peer learning."
    },
    {
        "title": "Wolfram Alpha — Computational Knowledge",
        "url": "https://www.wolframalpha.com",
        "subject": "Math",
        "level": "Intermediate",
        "description": "Computational intelligence engine for math, science, and engineering. Shows step-by-step solutions. Excellent for verifying work and understanding problem-solving approaches."
    },
]
