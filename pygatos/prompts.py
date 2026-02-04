"""LLM prompt templates for the GATOS workflow."""

# ============================================================================
# SUMMARIZATION PROMPTS
# ============================================================================

GENERIC_SUMMARY_SYSTEM = """You are an expert at summarizing text into concise bullet points.
Your task is to create a brief summary that captures the main points of the text.
Output 3-4 bullet points that summarize the key ideas."""

GENERIC_SUMMARY_PROMPT = """Summarize the following text into 3-4 brief bullet points:

{text}

Respond in JSON format:
{{
    "summary_points": [
        "First key point from the text",
        "Second key point from the text"
    ]
}}"""

GENERIC_SUMMARY_WITH_CONTEXT_PROMPT = """Here is context from the previous section:
{context}

Now summarize the following text into 3-4 brief bullet points:

{text}

Respond in JSON format:
{{
    "summary_points": [
        "First key point from the text",
        "Second key point from the text"
    ]
}}"""

INFORMATION_EXTRACTION_SYSTEM = """You are an expert qualitative researcher extracting key information from text.
Your task is to identify distinct, codable ideas expressed in the text.
Each information point should represent a single atomic idea that could be coded/categorized.
Be specific and capture the actual content, not meta-descriptions."""

INFORMATION_EXTRACTION_PROMPT = """Extract the key information points from the following text. Each point should represent a single, distinct idea that could be coded or categorized in qualitative analysis.

Text:
{text}

Respond in JSON format:
{{
    "information_points": [
        "First distinct idea expressed in the text",
        "Second distinct idea expressed in the text"
    ]
}}

Be specific and capture the actual content, not meta-descriptions. If the text expresses only one idea, return a single item in the array."""

INFORMATION_EXTRACTION_WITH_CONTEXT_PROMPT = """Here is context from the previous section:
{context}

Now extract the key information points from the following text. Each point should represent a single, distinct idea that could be coded or categorized in qualitative analysis.

Text:
{text}

Respond in JSON format:
{{
    "information_points": [
        "First distinct idea expressed in the text",
        "Second distinct idea expressed in the text"
    ]
}}

Be specific and capture the actual content, not meta-descriptions. Avoid repeating ideas already covered in the context."""

# ============================================================================
# CODE SUGGESTION PROMPTS
# ============================================================================

CODE_SUGGESTION_SYSTEM = """You are an expert qualitative researcher creating codes for thematic analysis.
A "code" is a short label that captures a recurring pattern or theme in the data.
Each code should have a clear name and definition.

IMPORTANT GUIDELINES:
- Prefer fewer, more encompassing codes over many narrow codes
- Avoid suggesting codes that overlap significantly in meaning
- Each code should capture a distinct concept, not just a different phrasing of the same idea
- If all excerpts share one common theme, suggest just one code
- Only suggest multiple codes if the excerpts clearly contain multiple distinct themes"""

CODE_SUGGESTION_PROMPT = """Below is a cluster of related text excerpts from qualitative data. Your task is to suggest codes that describe the theme(s) in this cluster.

Text excerpts:
{texts}

INSTRUCTIONS:
1. First, identify the core theme(s) present across these excerpts
2. Suggest the minimum number of codes needed to capture distinct themes
3. Do NOT suggest multiple codes that would apply to the same excerpts
4. If the excerpts share one primary theme, suggest just one well-defined code

For each code, provide:
- A short name (2-5 words)
- A clear definition (1-2 sentences) that distinguishes it from related concepts

Respond in JSON format:
{{
    "codes": [
        {{"name": "Code Name", "definition": "Definition of what this code represents and when to apply it."}}
    ]
}}"""

# ============================================================================
# NOVELTY EVALUATION PROMPTS
# ============================================================================

# Version 1: Original prompt (more permissive)
NOVELTY_EVALUATION_SYSTEM = """You are an expert qualitative researcher evaluating whether a proposed code is novel or redundant.
A code is REDUNDANT if it captures the same concept as an existing code, even if worded differently.
A code is NOVEL if it represents a distinct concept not already covered.

IMPORTANT: You must think through your reasoning BEFORE making your final decision. This prevents post-hoc rationalization."""

NOVELTY_EVALUATION_PROMPT = """Evaluate whether the following proposed code is novel or redundant compared to existing codes in the codebook.

PROPOSED CODE:
Name: {code_name}
Definition: {code_definition}

EXISTING SIMILAR CODES (most similar from codebook):
{existing_codes}

Evaluate step by step:
1. What is the core concept of the proposed code?
2. Does any existing code already capture this same concept, even if worded differently?
3. Is there a meaningful distinction that justifies adding this as a new code?

IMPORTANT: Write your reasoning FIRST, then make your decision based on that reasoning.

Respond in JSON format:
{{
    "similar_to": "Name of most similar existing code, or null if none",
    "reasoning": "Your step-by-step analysis comparing the proposed code to existing codes",
    "is_novel": true/false
}}"""

# Version 2: Stricter prompt with explicit criteria
NOVELTY_EVALUATION_SYSTEM_V2 = """You are an expert qualitative researcher evaluating whether a proposed code should be added to a codebook.
Your goal is to build a PARSIMONIOUS codebook—one with enough codes to capture distinct concepts, but not so many that codes overlap significantly.

REJECTION CRITERIA (reject if ANY apply):
1. SEMANTIC DUPLICATE: The proposed code means the same thing as an existing code, just with different wording.
2. SUBSET: The proposed code is a specific instance of a broader existing code (e.g., "Economic Anxiety" is a subset of "Economic Concerns").
3. NEAR-SYNONYM: The codes would be applied to nearly identical text excerpts in practice.

ACCEPTANCE CRITERIA (accept only if ALL apply):
1. DISTINCT CONCEPT: The proposed code captures a meaningfully different idea than all existing codes.
2. DIFFERENT APPLICATION: The proposed code would be applied to different text excerpts than existing codes.
3. ANALYTICAL VALUE: Adding this code would reveal patterns that existing codes cannot capture.

When in doubt, REJECT. It is better to have a focused codebook than a bloated one."""

NOVELTY_EVALUATION_PROMPT_V2 = """Evaluate whether the following proposed code should be added to the codebook.

PROPOSED CODE:
Name: {code_name}
Definition: {code_definition}

EXISTING SIMILAR CODES (most similar from codebook):
{existing_codes}

ANALYSIS STEPS:
1. Core concept: What specific idea does the proposed code capture?
2. Overlap check: For each existing code, would they be applied to the same or different text excerpts?
3. Redundancy test: If I removed the proposed code, would I lose the ability to capture an important distinction?

IMPORTANT: Complete your analysis FIRST, then make your decision based on that analysis.

Respond in JSON format:
{{
    "similar_to": "Name of most similar existing code, or null if none",
    "reasoning": "Your step-by-step analysis following the steps above",
    "is_novel": true/false
}}"""

# ============================================================================
# THEME GENERATION PROMPTS
# ============================================================================

THEME_SUGGESTION_SYSTEM = """You are an expert qualitative researcher organizing codes into themes.
A "theme" is a higher-level category that groups related codes together.
Themes should be broader than individual codes but still meaningful."""

THEME_SUGGESTION_PROMPT = """Below is a cluster of related codes from a qualitative codebook. Your task is to suggest a theme that would encompass these codes.

Codes in this cluster:
{codes}

Suggest a theme that groups these codes together:
1. A short theme name (2-5 words)
2. A clear definition (1-2 sentences)

Respond in JSON format:
{{
    "name": "Theme Name",
    "definition": "Definition of what this theme represents."
}}"""

# ============================================================================
# CODE APPLICATION PROMPTS
# ============================================================================

CODE_APPLICATION_SYSTEM = """You are an expert qualitative researcher applying codes to text data.
Your task is to determine which codes from a codebook apply to a given text excerpt.
Be precise: only select codes that clearly match the content."""

CODE_APPLICATION_PROMPT = """Determine which of the following codes apply to the text excerpt below.

TEXT EXCERPT:
{text}

CANDIDATE CODES:
{codes}

Select only the codes that clearly apply to this text. A code applies if the text expresses the concept described in the code's definition.

Respond in JSON format:
{{
    "applied_codes": ["Code Name 1", "Code Name 2"],
    "reasoning": "Brief explanation of why these codes apply"
}}

If no codes apply, return an empty list for applied_codes."""

# Version 2: Reasoning-first prompt to prevent post-hoc rationalization
CODE_APPLICATION_SYSTEM_V2 = """You are an expert qualitative researcher applying codes to text data.
Your task is to carefully analyze text and determine which codes from a codebook apply.

IMPORTANT: You must analyze the text BEFORE deciding which codes apply. This prevents post-hoc rationalization.

Apply a code only if the text CLEARLY expresses the concept described in the code's definition.
Be conservative: when in doubt, do NOT apply the code."""

CODE_APPLICATION_PROMPT_V2 = """Analyze the following text and determine which codes apply.

TEXT EXCERPT:
{text}

CANDIDATE CODES:
{codes}

ANALYSIS STEPS (complete in order):
1. Summarize: What is the text actually saying? What are the key ideas expressed?
2. Analyze: For each candidate code, does the text clearly express that concept?
3. Decide: Which codes should be applied based on your analysis?

IMPORTANT: Complete your analysis FIRST, then list the applied codes based on that analysis.

Respond in JSON format:
{{
    "text_summary": "Brief summary of what the text is expressing (1-2 sentences)",
    "analysis": "Your analysis comparing the text content to the candidate codes",
    "applied_codes": ["Code Name 1", "Code Name 2"]
}}

If no codes apply, return an empty list for applied_codes."""

# Version 3: Information point coding with source context
CODE_APPLICATION_SYSTEM_V3 = """You are an expert qualitative researcher applying codes to text data.
Your task is to carefully analyze an extracted information point and determine which codes apply.

You will be given:
1. An INFORMATION POINT - a specific atomic idea extracted from a larger response
2. SOURCE CONTEXT - the original text from which the information point was extracted

Focus your coding on the INFORMATION POINT, using the source context only to better understand its meaning.

IMPORTANT: You must analyze the information point BEFORE deciding which codes apply.
Apply a code only if the information point CLEARLY expresses the concept described in the code's definition.
Be conservative: when in doubt, do NOT apply the code."""

CODE_APPLICATION_PROMPT_V3 = """Analyze the following information point and determine which codes apply.

INFORMATION POINT:
{information_point}

SOURCE CONTEXT (for reference):
{source_text}

CANDIDATE CODES:
{codes}

ANALYSIS STEPS (complete in order):
1. Understand: What specific idea does this information point express?
2. Analyze: For each candidate code, does this information point clearly express that concept?
3. Decide: Which codes should be applied based on your analysis?

IMPORTANT: Focus on coding the INFORMATION POINT, not the entire source context.
Complete your analysis FIRST, then list the applied codes based on that analysis.

Respond in JSON format:
{{
    "point_interpretation": "Your interpretation of what this information point is expressing",
    "analysis": "Your analysis comparing the information point to the candidate codes",
    "applied_codes": ["Code Name 1", "Code Name 2"]
}}

If no codes apply, return an empty list for applied_codes."""

# ============================================================================
# QUESTION CANONICALIZATION PROMPTS
# ============================================================================

QUESTION_CANONICAL_SYSTEM = """You are an expert at identifying when questions are asking the same thing in different ways.
Your task is to analyze clusters of semantically similar questions and create canonical (standardized) versions.

IMPORTANT: Sometimes a cluster may contain questions that are related but actually asking different things.
In such cases, you should identify the distinct question intents and create separate canonical questions for each."""

QUESTION_CANONICAL_PROMPT = """Below is a cluster of questions that are semantically similar. They may be phrased differently but are asking about the same topic.

Questions in this cluster:
{questions}

ANALYSIS STEPS:
1. Read through all questions carefully
2. Identify how many DISTINCT question intents are present (usually 1, but sometimes 2-3)
3. For each distinct intent, create a canonical question

Create one or more canonical questions. Each canonical question should be:
- Clear and concise
- Representative of the questions it covers
- Neutral in phrasing

Respond in JSON format:
{{
    "analysis": "Brief explanation of how many distinct intents you found and why",
    "canonical_questions": [
        {{
            "canonical_question": "The standardized question",
            "topic": "Brief 2-4 word topic label",
            "covers_questions": [1, 2, 3]
        }}
    ]
}}

The "covers_questions" field should list the 1-indexed question numbers from the cluster that this canonical question represents.
If all questions ask the same thing, return a single canonical question covering all of them.
If you find 2-3 distinct intents, return multiple canonical questions, each covering a subset."""

QUESTION_ASSIGNMENT_SYSTEM = """You are an expert at matching questions to their canonical forms.
Your task is to determine which canonical question best represents the original question, or determine that none are a good match."""

QUESTION_ASSIGNMENT_PROMPT = """Given an original question from a focus group, select the best matching canonical question from the candidates below.

Original question:
{original_question}

Candidate canonical questions:
{candidates}

INSTRUCTIONS:
1. Read the original question carefully
2. Compare it to each candidate canonical question
3. Select the candidate that asks essentially the same thing (even if phrased differently)
4. If none of the candidates are a good match, indicate "no_match"

Respond in JSON format:
{{
    "reasoning": "Brief explanation of why you chose this match (or why none matched)",
    "selected_index": 1,
    "confidence": "high/medium/low"
}}

If no candidate is a good match, use:
{{
    "reasoning": "Explanation of why none of the candidates match",
    "selected_index": null,
    "confidence": "high"
}}

The "selected_index" should be the 1-indexed number of the best matching candidate, or null if no good match."""

# ============================================================================
# QUESTION CODEBOOK PROMPTS (Full GATOS Pipeline for Questions)
# ============================================================================

QUESTION_INTENT_EXTRACTION_SYSTEM = """You are an expert at understanding focus group questions.
Your task is to extract the core intent of what a question is asking about, independent of how it's phrased.
Focus on WHAT information the question seeks, not HOW it's worded.

For example:
- "What do you think of Trump?" and "How do you feel about Trump?" both have intent: "Opinion/evaluation of Donald Trump"
- "If the election were today, who would you vote for?" and "Who would you vote for if the election were held today?" both have intent: "Hypothetical current voting preference"
"""

QUESTION_INTENT_EXTRACTION_PROMPT = """Extract the core intent(s) of this focus group question. What is it really asking about?

Question:
{text}

Respond in JSON format:
{{
    "information_points": [
        "First core intent",
        "Second core intent (if the question asks about multiple distinct things)"
    ]
}}

IMPORTANT: Many focus group questions ask about MULTIPLE things. Extract EACH distinct intent as a separate item.

Guidelines:
- Strip away phrasing variations ("What do you think of...", "How do you feel about...", "Can you tell me...")
- Focus on the SUBJECT/TOPIC and the TYPE OF INFORMATION being sought
- Be concise but specific (e.g., "Opinion of Donald Trump as presidential candidate" not just "Trump")
- If the question asks about multiple distinct things, extract EACH as a separate intent
- Use a consistent format: "[Type of information] about/regarding [subject]"

Examples:
- "What do you think about Biden's handling of the economy?"
  → ["Evaluation of Biden's economic performance"]

- "Who would you vote for if the election were today?"
  → ["Hypothetical current presidential voting preference"]

- "How are things going in the country right now?"
  → ["Assessment of current national conditions"]

- "What do you think of voting by mail in general, and what do you think about voting by mail in the context of a pandemic?"
  → ["General opinion on voting by mail", "Opinion on voting by mail during a pandemic"]

- "Do you think Trump is doing a good job, and do you think he'll win in November?"
  → ["Evaluation of Trump's job performance", "Prediction of Trump's election chances"]
"""

QUESTION_CODE_SUGGESTION_SYSTEM = """You are an expert qualitative researcher creating codes for focus group questions.
A "code" represents a category of questions that ask about the same topic or seek the same type of information.

CRITICAL RULE - SEPARATE CODES FOR DIFFERENT ENTITIES:
- If the cluster contains questions about DIFFERENT PEOPLE, create SEPARATE codes for each person
- If the cluster contains questions about DIFFERENT TOPICS, create SEPARATE codes for each topic
- NEVER create a generic umbrella code like "Political Figure Opinion" when you can create specific codes like "Trump Opinion", "Biden Opinion", etc.

Example: If you see these intents in a cluster:
  - "Opinion of Paul Ryan"
  - "Opinion of Scott Walker"
  - "Opinion of Donald Trump"

WRONG: Create one code "Political Figure Opinion"
RIGHT: Create three codes: "Paul Ryan Opinion", "Scott Walker Opinion", "Trump Opinion"

IMPORTANT GUIDELINES:
- Code names should be CONCRETE and SPECIFIC to the actual topic being asked about
- Always include the specific person's name, policy area, or topic in the code name
- Avoid overly abstract or academic-sounding code names
- Keep codes grounded in what the questions actually ask about

GOOD code names (concrete, searchable):
- "Trump Job Performance" (not "Candidate Evaluation")
- "Paul Ryan Opinion" (not "Political Figure Opinion")
- "2024 Presidential Vote Choice" (not "Electoral Preferences")
- "COVID Response Criticism" (not "Pandemic Policy Assessment")
- "Biden Running Mate Importance" (not "Vice Presidential Selection Factors")

BAD code names (too abstract/generic):
- "Political Figure Opinion" - too generic, specify WHICH figure
- "Candidate Evaluation" - too generic, specify WHICH candidate
- "Perceived Factors Influencing Outcomes" - too vague
- "Attitudinal Dispositions Toward Governance" - too academic
"""

QUESTION_CODE_SUGGESTION_PROMPT = """Below are focus group question intents that have been grouped together because they ask about similar topics.

Question intents:
{texts}

Suggest codes to categorize these question types. Each code should be SPECIFIC to what the questions actually ask about.

IMPORTANT: If the intents mention DIFFERENT people (e.g., Trump, Biden, Paul Ryan), create a SEPARATE code for EACH person. Do NOT create a generic "Political Figure Opinion" code.

Respond in JSON format:
{{
    "codes": [
        {{
            "name": "Short, concrete code name (2-5 words)",
            "definition": "What specific questions this code covers"
        }}
    ]
}}

Guidelines:
- Create SEPARATE codes for different people/entities (e.g., "Trump Opinion" and "Biden Opinion", not "Political Figure Opinion")
- If all intents ask about the exact same thing, suggest just ONE code
- Code names should be CONCRETE - include the specific person's name or topic
- Avoid generic umbrella codes like "Political Figure Opinion" or "Candidate Evaluation"
- A good test: Would someone searching for "questions about Paul Ryan" find this code?
- The definition should be 1-2 sentences max"""

# ============================================================================
# QUESTION NOVELTY EVALUATION PROMPTS
# ============================================================================

QUESTION_NOVELTY_EVALUATION_SYSTEM = """You are an expert qualitative researcher evaluating whether a proposed question code should be added to a codebook.
Your goal is to build a PARSIMONIOUS codebook—one with enough codes to capture distinct question types, but not so many that codes overlap significantly.

CRITICAL: Codes about DIFFERENT ENTITIES are NEVER duplicates.
- "Trump Job Performance" and "Biden Job Performance" are DIFFERENT codes (different people)
- "Evan McMullin Supporter Profile" and "Gary Johnson Supporter Profile" are DIFFERENT codes (different candidates)
- "Republican Party Opinion" and "Democratic Party Opinion" are DIFFERENT codes (different parties)
- "COVID Policy" and "Immigration Policy" are DIFFERENT codes (different policy areas)

REJECTION CRITERIA (reject ONLY if the codes are about the SAME entity/topic):
1. SEMANTIC DUPLICATE: The proposed code asks about the same person/topic/thing as an existing code, just with different wording.
2. SUBSET: The proposed code is a specific instance of a broader existing code about the same entity.
3. NEAR-SYNONYM: The codes would be applied to nearly identical questions in practice.

ACCEPTANCE CRITERIA (accept if ANY apply):
1. DIFFERENT ENTITY: The proposed code is about a different person, party, candidate, or topic than existing codes.
2. DISTINCT CONCEPT: The proposed code captures a meaningfully different type of question.
3. DIFFERENT APPLICATION: The proposed code would be applied to different questions than existing codes.

When codes are about DIFFERENT entities, ALWAYS ACCEPT. Only reject when codes genuinely overlap on the SAME topic."""

QUESTION_NOVELTY_EVALUATION_PROMPT = """Evaluate whether the following proposed question code should be added to the codebook.

PROPOSED CODE:
Name: {code_name}
Definition: {code_definition}

EXISTING SIMILAR CODES (most similar from codebook):
{existing_codes}

ANALYSIS STEPS:
1. Entity check: What specific entity (person, party, topic) is the proposed code about?
2. Comparison: Is any existing code about the SAME entity/topic, or are they about different entities?
3. Overlap test: Would these codes be applied to the SAME questions, or different questions?

CRITICAL RULE: If the proposed code is about a DIFFERENT entity (e.g., different candidate, different policy area) than all existing codes, it MUST be accepted regardless of structural similarity.

IMPORTANT: Complete your analysis FIRST, then make your decision based on that analysis.

Respond in JSON format:
{{
    "similar_to": "Name of most similar existing code, or null if none",
    "reasoning": "Your step-by-step analysis following the steps above",
    "is_novel": true/false
}}"""

# ============================================================================
# QUESTION CODE APPLICATION PROMPTS
# ============================================================================

QUESTION_CODE_APPLICATION_SYSTEM = """You are an expert qualitative researcher applying topic codes to focus group question intents.
Your task is to determine which question topic codes match the intent of a question.

IMPORTANT: Be INCLUSIVE rather than conservative. If the question intent is about the same TOPIC as a code, apply that code.

Examples of matches:
- Intent: "Opinion about Biden's VP selection" → Code: "Biden VP Pick Importance" ✓ MATCH
- Intent: "Evaluation of Trump's handling of COVID" → Code: "Trump COVID-19 Handling" ✓ MATCH
- Intent: "What people think about Paul Ryan" → Code: "Paul Ryan Opinion" ✓ MATCH

Apply a code if the question intent is asking about the same topic, person, or issue that the code covers.
When in doubt, APPLY the code - it's better to over-apply than to miss relevant matches."""

QUESTION_CODE_APPLICATION_PROMPT = """Determine which codes apply to this question intent.

QUESTION INTENT:
{information_point}

ORIGINAL QUESTION (for context):
{source_text}

CANDIDATE CODES:
{codes}

For each candidate code, ask: "Is this question asking about the topic/person/issue that this code covers?"

Respond in JSON format:
{{
    "point_interpretation": "What topic/person/issue is this question asking about?",
    "analysis": "Which codes cover this topic? (Be inclusive - if the question is about the code's topic, apply it)",
    "applied_codes": ["Code Name 1", "Code Name 2"]
}}

IMPORTANT: Apply codes generously. If the question is about Biden's running mate, apply "Biden VP Pick Importance". If it's about Trump, apply the relevant Trump code."""

# ============================================================================
# STARTER CODES PROMPTS
# ============================================================================

STARTER_CODES_SYSTEM = """You are an expert qualitative researcher familiar with common themes in research studies.
Your task is to generate hypothetical codes that might appear in a study on a given topic."""

STARTER_CODES_PROMPT = """Generate {n_codes} hypothetical codes that one might expect to find in a qualitative study about:

Topic: {topic}

These codes will be used as "starter codes" to help initialize a codebook generation process.
Each code should be distinct and represent a plausible theme that might emerge from data on this topic.

Respond in JSON format:
{{
    "codes": [
        {{"name": "Code Name", "definition": "Definition of what this code represents."}}
    ]
}}"""

# ============================================================================
# ANSWER EXTRACTION PROMPTS
# ============================================================================
# These prompts are for extracting information points from focus group answers.
# They provide explicit context that the text is a participant's response to a question.

ANSWER_EXTRACTION_SYSTEM = """You are an expert qualitative researcher extracting key information from focus group answers.
Your task is to identify distinct, codable ideas expressed by the participant in their response.

Each information point should represent a single atomic idea that could be coded/categorized:
- An opinion or attitude the participant holds
- A reason or justification they give
- An experience or observation they describe
- A belief or perception they express

IMPORTANT: The participant is responding to a specific question. Extract what they are saying IN RESPONSE TO that question.
Focus on the CONTENT of their answer - what they think, feel, believe, or have experienced.
Be specific and capture the actual substance, not meta-descriptions like "participant expressed an opinion"."""

ANSWER_EXTRACTION_PROMPT = """The following is a participant's answer from a focus group discussion.

QUESTION ASKED:
{question}

PARTICIPANT'S ANSWER:
{answer}

Extract the key information points from the participant's answer. Each point should represent a single, distinct idea that could be coded - an opinion, reason, experience, or belief.

Respond in JSON format:
{{
    "information_points": [
        "First distinct idea expressed by the participant",
        "Second distinct idea expressed by the participant"
    ]
}}

Guidelines:
- Focus on WHAT the participant is saying, thinking, or feeling
- Each point should be specific and substantive
- Preserve the participant's perspective (e.g., "Believes Trump handled COVID poorly" not "Opinion on COVID response")
- If the answer is short or expresses only one idea, return a single item
- Avoid meta-descriptions - capture the actual content"""

ANSWER_EXTRACTION_WITH_CONTEXT_PROMPT = """The following is a participant's answer from a focus group discussion.

QUESTION ASKED:
{question}

Here is context from the previous section of this answer:
{context}

CURRENT SECTION OF ANSWER:
{answer}

Extract the key information points from this section. Each point should represent a single, distinct idea.
Avoid repeating ideas already covered in the context.

Respond in JSON format:
{{
    "information_points": [
        "First distinct idea in this section",
        "Second distinct idea in this section"
    ]
}}"""

# ============================================================================
# ANSWER CODE SUGGESTION PROMPTS
# ============================================================================
# These prompts are for suggesting codes from clustered answer content.

ANSWER_CODE_SUGGESTION_SYSTEM = """You are an expert qualitative researcher creating codes for focus group answer content.
A "code" represents a recurring pattern in how participants respond - a type of opinion, reasoning, experience, or attitude.

Your codes should capture WHAT PARTICIPANTS ARE SAYING, not the topic of the question they're answering.

CRITICAL RULE: Create ENTITY-SPECIFIC codes when responses are about specific people, parties, or policies.
- WRONG: "Political Figure Assessment" (too generic)
- RIGHT: "Negative Assessment of Biden", "Positive Assessment of Trump" (entity-specific)
- WRONG: "Policy Opinion" (too generic)
- RIGHT: "COVID Restriction Opposition", "Immigration Policy Support" (policy-specific)

GOOD code names capture participant responses WITH specificity:
- "Distrust of Biden Competence" - participants expressing doubts about Biden specifically
- "Personal COVID Impact" - participants describing how COVID affected them personally
- "Trust in Trump Leadership" - participants expressing confidence in Trump specifically
- "Media Distrust" - participants expressing skepticism of news sources
- "Party Loyalty Conflict" - participants describing tension between party and candidate

BAD code names (too generic or vague):
- "COVID Question Response" - doesn't say what they responded
- "Political Figure Opinion" - too generic, which figure?
- "Political Views" - too generic
- "Candidate Assessment" - which candidate?

IMPORTANT: Create codes that capture the NATURE or SUBSTANCE of responses, including WHO or WHAT they are about."""

ANSWER_CODE_SUGGESTION_PROMPT = """Below are excerpts from focus group participants' answers that have been grouped together because they express similar ideas.

Answer excerpts:
{texts}

Suggest codes that capture the patterns in these responses. Each code should represent a type of opinion, reasoning, experience, or attitude expressed by participants.

Respond in JSON format:
{{
    "codes": [
        {{
            "name": "Short, descriptive code name (2-5 words)",
            "definition": "What type of participant response this code captures"
        }}
    ]
}}

Guidelines:
- Codes should describe WHAT participants are saying/feeling/believing
- If responses are about a SPECIFIC person/party/policy, include that in the code name
- If all excerpts express the same type of response, suggest just ONE code
- Code names should be specific enough that a researcher knows what responses to expect
- Definitions should be 1-2 sentences describing the pattern of responses this code captures
- AVOID generic codes like "Political Figure Assessment" - use "Biden Disapproval" or "Trump Approval" instead"""

# ============================================================================
# ANSWER NOVELTY EVALUATION PROMPTS
# ============================================================================
# These prompts evaluate whether a proposed answer code should be added to the codebook.

ANSWER_NOVELTY_EVALUATION_SYSTEM = """You are an expert qualitative researcher evaluating whether a proposed code should be added to an answer codebook.
Your goal is to build a PARSIMONIOUS codebook - enough codes to capture distinct response patterns, but not so many that codes overlap.

CRITICAL: Codes about DIFFERENT ENTITIES are NEVER duplicates.
- "Negative Assessment of Biden" and "Negative Assessment of Trump" are DIFFERENT codes (different people)
- "Trust in Federal Government" and "Trust in State Government" are DIFFERENT codes (different levels of government)
- "Economic Pessimism" and "Healthcare Pessimism" are DIFFERENT codes (different policy areas)
- "Praise for Biden Leadership" and "Evaluations of Political Figures" are DIFFERENT codes (specific vs. generic)

REJECTION CRITERIA (reject ONLY if codes are about the SAME entity/topic):
1. SEMANTIC DUPLICATE: The proposed code captures the same response pattern about the SAME entity as an existing code
2. SUBSET: The proposed code is a narrower version of an existing code about the SAME entity
3. NEAR-SYNONYM: Researchers would apply both codes to the SAME participant responses about the SAME topic

ACCEPTANCE CRITERIA (accept if ANY apply):
1. DIFFERENT ENTITY: The proposed code is about a different person, party, policy area, or topic
2. DIFFERENT VALENCE: The proposed code captures an opposite stance (e.g., "Biden Approval" vs "Biden Disapproval")
3. SPECIFIC vs GENERIC: The proposed code is about a specific entity while existing code is generic (always keep the specific one)
4. DISTINCT PATTERN: The proposed code captures a different type of response

When codes are about DIFFERENT entities, ALWAYS ACCEPT. Only reject when codes genuinely overlap on the SAME topic/entity."""

ANSWER_NOVELTY_EVALUATION_PROMPT = """Evaluate whether the following proposed code should be added to the answer codebook.

PROPOSED CODE:
Name: {code_name}
Definition: {code_definition}

EXISTING SIMILAR CODES (most similar from codebook):
{existing_codes}

ANALYSIS STEPS:
1. Entity check: What specific entity (person, party, policy area) is the proposed code about?
2. Comparison: Is any existing code about the SAME entity, or are they about different entities?
3. Specificity check: Is the proposed code more specific than a generic existing code?
4. Overlap test: Would these codes be applied to the SAME participant responses, or different responses?

CRITICAL RULE: If the proposed code is about a DIFFERENT entity (e.g., different politician, different policy area) than all existing codes, it MUST be accepted regardless of structural similarity.

IMPORTANT: A specific code about one entity (e.g., "Negative Assessment of Biden") should NOT be rejected in favor of a generic code (e.g., "Evaluations of Political Figures"). Specific codes capture more useful information.

Respond in JSON format:
{{
    "similar_to": "Name of most similar existing code, or null if none",
    "reasoning": "Your step-by-step analysis following the steps above",
    "is_novel": true/false
}}"""

# ============================================================================
# ANSWER CODE APPLICATION PROMPTS
# ============================================================================
# These prompts apply answer codes to participant responses.

ANSWER_CODE_APPLICATION_SYSTEM = """You are an expert qualitative researcher applying codes to focus group participant responses.
Your task is to determine which codes match the ideas expressed in a participant's answer.

IMPORTANT: Apply codes based on WHAT THE PARTICIPANT IS SAYING - their opinions, reasons, experiences, and attitudes.

A code APPLIES if the participant's response matches the pattern the code describes.
For example:
- If a participant says "I don't trust the news anymore" → "Media Distrust" applies
- If a participant says "The economy is terrible under Biden" → "Economic Pessimism" applies
- If a participant says "I voted for Trump because he's a businessman" → "Business Leadership Appeal" applies

Be INCLUSIVE: If the participant's response reflects the pattern a code describes, apply it.
When in doubt, APPLY the code - it's better to over-apply than miss relevant matches."""

ANSWER_CODE_APPLICATION_PROMPT = """Determine which codes apply to this participant response.

INFORMATION POINT (extracted from participant's answer):
{information_point}

ORIGINAL CONTEXT:
{source_text}

CANDIDATE CODES:
{codes}

For each candidate code, ask: "Does this participant's response match the pattern this code describes?"

Respond in JSON format:
{{
    "point_interpretation": "What is the participant expressing? (opinion, reason, experience, etc.)",
    "analysis": "Which codes match this type of response?",
    "applied_codes": ["Code Name 1", "Code Name 2"]
}}

Be inclusive - if the response reflects the pattern a code describes, apply it."""


def format_codes_for_prompt(codes: list, include_definition: bool = True) -> str:
    """Format a list of codes for inclusion in a prompt."""
    lines = []
    for i, code in enumerate(codes, 1):
        if hasattr(code, 'name'):
            name = code.name
            definition = code.definition if include_definition else ""
        else:
            name = code.get('name', code)
            definition = code.get('definition', '') if include_definition else ""

        if include_definition and definition:
            lines.append(f"{i}. {name}: {definition}")
        else:
            lines.append(f"{i}. {name}")

    return "\n".join(lines)


def format_texts_for_prompt(texts: list, max_texts: int = 20) -> str:
    """Format a list of texts for inclusion in a prompt."""
    if len(texts) > max_texts:
        texts = texts[:max_texts]
        truncated = True
    else:
        truncated = False

    lines = [f"- {text}" for text in texts]

    if truncated:
        lines.append(f"... and {len(texts) - max_texts} more")

    return "\n".join(lines)


def add_study_context(system_prompt: str, study_context: str | None) -> str:
    """Add study context to a system prompt if provided.

    This function appends study context information to system prompts,
    helping the LLM understand the domain and produce more relevant output.

    Args:
        system_prompt: The base system prompt.
        study_context: Optional context about the study/dataset.

    Returns:
        The system prompt with context appended, or the original if no context.

    Example:
        >>> base = "You are an expert qualitative researcher."
        >>> context = "Survey responses about inflation concerns from US adults."
        >>> result = add_study_context(base, context)
        >>> print(result)
        You are an expert qualitative researcher.

        STUDY CONTEXT:
        Survey responses about inflation concerns from US adults.
    """
    if not study_context:
        return system_prompt

    return f"""{system_prompt}

STUDY CONTEXT:
{study_context}"""
