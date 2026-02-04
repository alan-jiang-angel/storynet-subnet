"""
LLM Generator

Unified story generation backend supporting both local and cloud LLMs.

Local Mode:
  - Ollama (localhost:11434)
  - vLLM (OpenAI-compatible endpoint)

Cloud Mode:
  - OpenAI (GPT-4, GPT-4o-mini)
  - Google Gemini
  - Zhipu AI (GLM-4)
"""

import os
import time
import asyncio
import logging
from typing import Dict, Optional
from .base import StoryGenerator, GenerationError, GeneratorConfigError

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except ImportError:
#     GEMINI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class LLMGenerator(StoryGenerator):
    """
    Unified LLM generator for story generation.

    Supports two modes:
    - local: Ollama, vLLM, or any OpenAI-compatible local server
    - cloud: OpenAI, Gemini, Zhipu cloud APIs
    """

    def __init__(self, config: Dict):
        """
        Initialize LLM generator.

        Args:
            config: Configuration dict with mode-specific settings
        """
        super().__init__(config)

        self.mode = config.get("mode", "cloud")
        self.available = False
        self.initialized = False

        if self.mode == "local":
            self._init_local(config.get("local", {}))
        else:
            self._init_cloud(config.get("cloud", {}))

    def _init_local(self, config: Dict):
        """Initialize local LLM backend."""
        self.local_type = config.get("type", "ollama")
        self.local_url = config.get("url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5:7b")

        if self.local_type == "ollama":
            # Ollama uses its own API format
            self.api_endpoint = f"{self.local_url}/api/generate"
            self.use_chat_format = False
        else:
            # vLLM and others use OpenAI-compatible format
            self.api_endpoint = f"{self.local_url}/v1/chat/completions"
            self.use_chat_format = True

            if OPENAI_AVAILABLE:
                self.client = AsyncOpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=f"{self.local_url}/v1"
                )

        self.available = True
        self.initialized = True
        logger.info(f"Local LLM initialized: {self.local_type} @ {self.local_url}")

    def _init_cloud(self, config: Dict):
        """Initialize cloud LLM backend."""
        self.provider = config.get("provider", "openai")
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4o-mini")
        self.endpoint = config.get("endpoint")

        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            logger.warning(f"{api_key_env} not found in environment")
            return

        self.available = True

        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise GeneratorConfigError("openai library not installed")
            if self.endpoint:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.endpoint)
            else:
                self.client = AsyncOpenAI(api_key=self.api_key)

        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise GeneratorConfigError("google-generativeai not installed")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)

        elif self.provider == "zhipu":
            if not HTTPX_AVAILABLE:
                raise GeneratorConfigError("httpx not installed")
            self.zhipu_endpoint = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

        self.initialized = True
        logger.info(f"Cloud LLM initialized: {self.provider}/{self.model}")

    async def generate(self, input_data: Dict) -> Dict:
        """Generate story content."""
        if not self.available:
            raise GenerationError("LLM Generator not available")

        start_time = time.time()

        try:
            if self.mode == "local":
                result = await self._generate_local(input_data)
            else:
                result = await self._generate_cloud(input_data)

            generation_time = time.time() - start_time

            return {
                "generated_content": result,
                "model": self.model,
                "mode": self.mode,
                "generation_time": generation_time,
                "metadata": {
                    "type": self.local_type if self.mode == "local" else self.provider
                }
            }

        except Exception as e:
            raise GenerationError(f"Generation failed: {str(e)}")

    async def _generate_local(self, input_data: Dict) -> str:
        """Generate using local LLM."""
        if self.local_type == "ollama":
            return await self._generate_ollama(input_data)
        else:
            # vLLM or other OpenAI-compatible
            return await self._generate_openai_compatible(input_data)

    async def _generate_ollama(self, input_data: Dict) -> str:
        """Generate using Ollama API."""
        if not HTTPX_AVAILABLE:
            raise GenerationError("httpx not installed")

        prompt = self._build_prompt(input_data)

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 2048
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")

    async def _generate_openai_compatible(self, input_data: Dict) -> str:
        """Generate using OpenAI-compatible API (vLLM, etc.)."""
        messages = self._build_messages(input_data)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=2048
        )

        return response.choices[0].message.content

    async def _generate_cloud(self, input_data: Dict) -> str:
        """Generate using cloud API."""
        if self.provider == "openai":
            return await self._generate_openai(input_data)
        elif self.provider == "gemini":
            return await self._generate_gemini(input_data)
        elif self.provider == "zhipu":
            return await self._generate_zhipu(input_data)
        else:
            raise GenerationError(f"Unsupported provider: {self.provider}")

    async def _generate_openai(self, input_data: Dict) -> str:
        """Generate using OpenAI API."""
        messages = self._build_messages(input_data)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=2048
        )

        return response.choices[0].message.content

    async def _generate_gemini(self, input_data: Dict) -> str:
        """Generate using Gemini API."""
        prompt = self._build_prompt(input_data)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.client.generate_content,
            prompt
        )

        return response.text

    async def _generate_zhipu(self, input_data: Dict) -> str:
        """Generate using Zhipu API."""
        messages = self._build_messages(input_data)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.zhipu_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.8,
                    "max_tokens": 2048
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _build_messages(self, input_data: Dict) -> list:
        task_type = input_data.get("task_type")
        user_input = input_data.get("user_input", "")
        blueprint = input_data.get("blueprint", {})
        characters = input_data.get("characters", [])
        story_arc = input_data.get("story_arc", {})
        chapters = input_data.get("chapters", [])
        chapter_ids = input_data.get("chapter_ids", [])
        random_uuid_or_timestamp = int(time.time())

        if (task_type == 'blueprint'):
            system_prompt = "You are a senior story architect designing a STORY BLUEPRINT for a long-form narrative."
            content = f"""
Your output will be automatically evaluated by multiple scoring systems (technical, structural, content, and narrative quality). You MUST follow these rules exactly to maximize your score.

CRITICAL FORMAT REQUIREMENTS
- Respond with ONE valid JSON OBJECT only.
- TOP-LEVEL TYPE: JSON OBJECT `{ ... }` (NOT an array).
- Do NOT include explanations, comments, markdown, or code fences.
- Use double quotes for all keys and string values.
- Do NOT include trailing commas.
- The JSON MUST be directly parseable by a strict JSON parser.

REQUIRED TOP-LEVEL FIELDS (all required, no nulls, no empty strings):
- "title": string
- "genre": string
- "setting": string
- "core_conflict": string
- "themes": array of strings
- "tone": string
- "target_audience": string
- "format": "text",
- "task_type": "blueprint"

USER REQUEST (story idea to base everything on):
"${user_input}"

1) TECHNICAL & SCHEMA REQUIREMENTS (HIGH PRIORITY)
==================================================
To pass technical and schema checks, strictly ensure:

- All required fields exist at the top level:
  - "title"
  - "genre"
  - "setting"
  - "core_conflict"
  - "themes"
  - "tone"
  - "target_audience"

- Field types:
  - "title": non-empty string.
  - "genre": non-empty string.
  - "setting": non-empty string.
  - "core_conflict": non-empty string.
  - "themes": JSON ARRAY of strings, length between 2 and 5 INCLUSIVE.
    - Example: "themes": ["Redemption", "Surveillance vs freedom", "Family loyalty"]
  - "tone": non-empty string.
  - "target_audience": non-empty string.

- "themes" rules:
  - Must be an array, not a single string.
  - Must contain 2–5 distinct, concise theme phrases.
  - Each element must be a string (no objects or numbers).

If any of these are missing or wrong, the technical score will collapse.

2) STRUCTURE & LENGTH (BLUEPRINT-SPECIFIC)
===========================================
For the structure score, the blueprint must be complete and appropriately detailed:

A. Field completeness (very important)
- All required fields above must be present and feel meaningfully filled, not placeholders.

B. "setting" length and content
- Goal: A rich, concrete paragraph describing:
  - Time period, main locations, social/technological context, atmosphere.
  - Elements that clearly align with the user request.
- Length guideline (characters):
  - Ideal: around 50–300 characters.
  - Absolutely avoid being too short or a single vague sentence.
- Make the setting specific and vivid, but still high-level enough to support many possible plots.

C. "core_conflict" length and content
- Goal: Clearly define the story-driving conflict:
  - Who is the main protagonist, what is their primary goal?
  - Who or what opposes them (antagonist, system, environment, inner conflict, etc.)?
  - What is at stake if they fail?
  - Why this conflict is difficult, morally complex, or surprising?
- Length guideline:
  - Aim for 30-200 characters.
  - Use several sentences with proper punctuation.

D. "themes" count
- Provide 2–5 themes (ideal: 3–4), each a short phrase.
- Themes should be conceptually distinct (e.g. “Memory and identity”, “Class inequality”, “Autonomy vs control”) rather than weak variants of the same word.

3) CONTENT RELEVANCE & FLUENCY SCORE
=====================================
These checks are heuristic and embedding-based; maximize them by:

A. Relevance to user input (very important)
- Heuristic relevance uses keyword overlap between the USER REQUEST and:
  - "title"
  - "genre"
  - "setting"
- Therefore:
  - Reuse important nouns and short phrases from "${user_input}" naturally in the title, genre, and especially in the setting.
  - If the user mentions specific elements (e.g., “cyberpunk hacker in a megacity”, “post-apocalyptic desert caravan”), echo these concepts and vocabulary in "genre" and "setting".
  - Do NOT just copy the input; integrate and expand it.

B. Fluency and readability
- Ensure "setting" and "core_conflict" are:
  - Grammatically correct and well-punctuated.
  - Written in natural, fluent prose with multiple sentences.
  - Free from obvious repetition or word salad.
- Aim for:
  - Diverse sentence structures (short and medium-length sentences mixed).
  - Clear logical flow: introduce context → highlight tension → suggest trajectory.

C. Reasonable overall length
- Combined, "setting" + "core_conflict" should have enough text to:
  - Exceed ~100 words easily.
  - Provide substance for evaluation, but avoid needlessly bloated paragraphs.

4) NARRATIVE MERIT & STORY CRAFT GOALS
=======================================
An additional AI evaluator will examine the narrative quality of the blueprint. Optimize for:

A. Narrative flow
- The implied story should have a clear beginning situation, escalating tension, and stakes hinted at in the conflict.
- Even though this is only a blueprint, the reader should sense a natural progression of events the story could follow.

B. Emotional resonance
- Build emotional hooks directly into "setting" and "core_conflict":
  - Personal stakes (relationships, identity, freedom, survival, justice, etc.).
  - Conflicting desires or values.
- Use specific details (“a lone courier smuggling outlawed memories across neon-drenched slums”) rather than abstract generalities (“a person faces challenges in a city”).

C. Creative originality
- Avoid copying known works, franchises, or famous IP.
- Add at least one distinctive twist or unusual combination:
  - E.g., unexpected setting mashups, moral dilemmas, or unique systems/powers.
- Themes should suggest depth (e.g. “Memory as currency” rather than just “technology”).

D. Internal consistency
- Make sure all fields agree:
  - The "title", "genre", "setting", "core_conflict", "themes", "tone", and "target_audience" must clearly describe the SAME story.
  - For example, do not pair a comedic tone with a purely grim, hopeless conflict unless you explicitly frame it as dark comedy.

5) CONSISTENCY OF GENRE, TONE, TARGET AUDIENCE FIELDS
====================================================
- "genre":
  - Use a clear, shelf-ready label or combination like:
    - "cyberpunk thriller"
    - "post-apocalyptic survival drama"
    - "historical fantasy mystery"
  - Let it meaningfully reflect the user’s request and the conflict.

- "tone":
  - A brief phrase such as:
    - "dark and suspenseful"
    - "bittersweet and introspective"
    - "tense yet hopeful"
  - Must match the emotional flavor suggested in "setting" and "core_conflict".

- "target_audience":
  - Keep it concise and specific:
    - "adult readers who enjoy character-driven cyberpunk mysteries"
    - "young adult readers who like emotionally intense portal fantasies"

6) ORIGINALITY SAFETY
====================
- DO NOT refer to or closely mimic existing copyrighted stories, movies, games, anime, or named franchises.
- Use original character names, locations, organizations, and conflicts tailored to "${user_input}".

7) FINAL OUTPUT SHAPE (STRICT EXAMPLE)
=========================================
Your final answer MUST be ONLY one JSON object in this exact structural shape (keys and types), with your own values filled in:

{{
  "title": "...",
  "genre": "...",
  "setting": "...",
  "core_conflict": "...",
  "themes": ["...", "..."],
  "tone": "...",
  "target_audience": "..."
}}

8) VARIATION SEED
========================================= 
${random_uuid_or_timestamp}
Use the variation seed only to influence creative decisions so that each run produces a substantially different story blueprint.
"""
        elif (task_type == 'characters'):
            system_prompt = "You are a professional character designer for an interactive story game."
            content = f"""

    Your output will be automatically evaluated by multiple scoring systems (technical, structural, content, and narrative quality). You MUST follow these rules exactly to maximize your score.

CRITICAL FORMAT REQUIREMENTS
- Respond with ONE valid JSON OBJECT only.
- TOP-LEVEL TYPE: JSON OBJECT `{{ ... }}` (NOT an array).
- Do NOT include explanations, comments, markdown, or code fences.
- Use double quotes for all keys and string values.
- Do NOT include trailing commas.
- The JSON MUST be directly parseable by a strict JSON parser.

REQUIRED TOP-LEVEL STRUCTURE:
{{
  "characters": [
    {{
      "id": "string",
      "name": "string",
      "archetype": "string",
      "background": "string",
      "motivation": "string",
      "skills": ["string"],
      "personality_traits": ["string"],
      "relationships": {{
        "character_id": "relationship description"
      }}
    }}
  ],
  "format": "text",
  "task_type": "characters"
}}

INPUT CONTEXT:
- Blueprint: "${blueprint}"
- User Input: "${user_input}"

Use BOTH sources when designing characters. If they conflict, prefer the user_input.


1) TECHNICAL & SCHEMA REQUIREMENTS (CRITICAL - 30 POINTS)
==================================================
To pass technical and schema checks, strictly ensure:

A. JSON Validity (10 points)
- Output MUST be valid JSON that can be parsed by `json.loads()`.
- No syntax errors, no unescaped quotes, no trailing commas.
- The root must be a JSON object `{{ ... }}`, NOT an array `[ ... ]`.

B. Schema Completeness (10 points)
- MUST have exactly 5 characters in the "characters" array (no more, no less).
- Each character MUST have the exact "id" values:
  - One character with "id": "protagonist"
  - One character with "id": "ally"
  - One character with "id": "rival"
  - One character with "id": "mentor"
  - One character with "id": "wildcard"
- These IDs are REQUIRED and must match exactly (case-sensitive).
- Each character object MUST be a valid JSON object (dict), not a string or null.

C. Response Time (10 points)
- This is not controllable by the prompt, but ensure your output is concise enough to generate quickly.

2) STRUCTURE REQUIREMENTS (40 POINTS TOTAL)
==================================================

A. Character Count (20 points)
- Generate EXACTLY 5 characters (not 4, not 6, exactly 5).
- If you generate fewer or more, you lose the full 20 points.

B. Character Completeness (10 points)
Each character MUST include ALL of these fields with non-empty values:
- "id": string (must be one of: "protagonist", "ally", "rival", "mentor", "wildcard")
- "name": string (character's name, non-empty)
- "archetype": string (e.g., "Hero", "Mentor", "Shadow", "Trickster", "Guardian")
- "background": string (detailed character history and context - see length requirements below)
- "motivation": string (what drives this character, their primary goal)
- "skills": array of strings (minimum 2 items, each a skill/ability)
- "personality_traits": array of strings (minimum 2 items, each a trait)

Missing any field = partial score deduction.

C. Relationships Network (10 points)
- Each character MUST have a "relationships" object (dict).
- Each "relationships" object MUST contain at least 2 entries (key-value pairs).
- Each key should be another character's "id" from your character list.
- Each value should be a string describing the relationship (e.g., "trusted friend", "bitter rival", "former mentor").
- Example:
  {{
    "relationships": {{
      "ally": "trusted companion who shares the protagonist's goals",
      "rival": "competitor who challenges the protagonist's methods"
    }}
  }}
- Having exactly 2 relationships per character = full score.
- Having only 1 relationship = partial score (5 points).
- Having 0 relationships = 0 points.

3) CONTENT QUALITY REQUIREMENTS (30 POINTS TOTAL)
==================================================

A. Relevance to Blueprint (15 points)
- The scoring system uses keyword overlap between the blueprint JSON and your characters JSON.
- To maximize relevance:
  - Reuse important nouns, adjectives, and phrases from the blueprint's fields:
    - "title", "genre", "setting", "core_conflict", "themes", "tone"
  - Incorporate blueprint vocabulary naturally into:
    - Character "background" fields (most important)
    - Character "name" fields (when appropriate)
    - Character "archetype" fields
    - Character "motivation" fields
  - Example: If blueprint has "setting": "cyberpunk megacity", use words like "cyberpunk", "megacity", "neon", "corporate", "hacker" in character backgrounds.
  - Example: If blueprint has "themes": ["Memory", "Identity"], reference memory and identity concepts in character backgrounds.
  - The more keyword overlap, the higher the relevance score (up to 15 points).

B. Fluency and Readability (10 points)
The scoring system checks:
- Punctuation presence: Use proper punctuation (periods, commas, question marks, exclamation marks) in "background" fields.
- Repetition ratio: Avoid repeating the same words excessively. Aim for unique word ratio > 60%.
- Word count: Combined "background" text from all 5 characters should exceed 100 words total.
- Sentence variety: Use varied sentence lengths (mix short and medium sentences).
- Average sentence length: Keep between 10-30 words per sentence.
- Total sentences: Have at least 3 sentences combined across all character backgrounds.

C. Originality (5 points)
- Avoid copying characters from existing stories, games, movies, or franchises.
- Create original character concepts that feel fresh and specific to the blueprint.
- Each character should have distinctive traits that aren't generic stereotypes.

4) NARRATIVE MERIT REQUIREMENTS (30 POINTS TOTAL)
==================================================
An AI evaluator will assess narrative quality. Optimize for:

A. Narrative Flow (weighted 30%)
- Characters should form a coherent ensemble that supports the blueprint's story.
- Character relationships should create natural dramatic tension and story progression.
- The five characters together should enable interesting conflicts, alliances, and plot developments.

B. Emotional Impact (weighted 25%)
- Characters should have clear emotional stakes and motivations.
- Include personal details in "background" that make characters relatable or compelling.
- Show how characters' goals and fears connect to the blueprint's core conflict.
- Use vivid, specific details rather than abstract descriptions.

C. Creative Originality (weighted 25%)
- Avoid generic archetypes unless the blueprint explicitly calls for them.
- Give each character a unique twist or unexpected quality.
- No two characters should share the same archetype (enforced by structure rules).
- Characters should feel fresh and not derivative of famous characters from media.

D. Internal Consistency (weighted 20%)
- All characters must fit the blueprint's:
  - Genre (e.g., fantasy characters in a fantasy setting)
  - Setting (e.g., characters appropriate to the time period and location)
  - Tone (e.g., serious characters in a serious story, comedic in a comedy)
  - Themes (characters should embody or contrast with the themes)
- Character relationships should make logical sense within the story world.
- Character skills and backgrounds should align with the setting's rules and constraints.

5) CHARACTER-SPECIFIC DESIGN GUIDELINES
==================================================

A. The Five Required Characters:

1. "protagonist" (id: "protagonist")
   - The main character who drives the story.
   - Should align with the blueprint's core conflict as the primary actor.
   - Background should be detailed and compelling (aim for 30-60 words).
   - Motivation should directly relate to the blueprint's core conflict.

2. "ally" (id: "ally")
   - A supporting character who helps the protagonist.
   - Should have complementary skills to the protagonist.
   - Background should explain why they support the protagonist (20-50 words).
   - At least one relationship should reference "protagonist" with a positive dynamic.

3. "rival" (id: "rival")
   - An opposing force or competitor.
   - Can be antagonist or friendly competitor.
   - Background should explain the source of conflict or competition (20-50 words).
   - At least one relationship should reference "protagonist" with a competitive or adversarial dynamic.

4. "mentor" (id: "mentor")
   - A guide or teacher figure.
   - Should have wisdom or experience relevant to the blueprint's themes.
   - Background should establish their expertise and why they mentor (20-50 words).
   - At least one relationship should reference "protagonist" with a guidance dynamic.

5. "wildcard" (id: "wildcard")
   - An unpredictable or mysterious character.
   - Should add complexity and unpredictability to the story.
   - Background should hint at hidden motives or secrets (20-50 words).
   - Relationships should be ambiguous or surprising.

B. Field-Specific Requirements:

- "background": 
  - Should be a rich paragraph (not a single sentence).
  - Include: origin, key life events, current situation, relevant history.
  - Length: Aim for 20-60 words per character (100+ words total across all 5).
  - Use proper punctuation and varied sentence structures.
  - Incorporate blueprint keywords naturally.

- "motivation":
  - Clear, specific goal or drive (not vague like "wants to succeed").
  - Should connect to the blueprint's core conflict.
  - Length: 10-30 words.

- "skills":
  - Minimum 2 items, each a concrete skill or ability.
  - Should be relevant to the blueprint's setting and genre.
  - Examples: ["Hacking", "Stealth", "Negotiation"] or ["Magic", "Swordsmanship", "Healing"]

- "personality_traits":
  - Minimum 2 items, each a distinct personality characteristic.
  - Should be specific (not generic like "nice" or "brave").
  - Examples: ["Cautious", "Loyal", "Quick-tempered"] or ["Curious", "Skeptical", "Impulsive"]

- "relationships":
  - Minimum 2 entries per character.
  - Keys must be valid character IDs from your list ("protagonist", "ally", "rival", "mentor", "wildcard").
  - Values should be descriptive strings explaining the relationship dynamic.
  - Ensure bidirectional consistency where logical (if A trusts B, B's relationship to A should reflect trust or respect).

6) BLUEPRINT INTEGRATION STRATEGY
==================================================

To maximize relevance score (15 points), systematically integrate blueprint elements:

1. Extract key terms from blueprint:
   - From "title": Extract nouns and distinctive words.
   - From "genre": Extract genre-specific vocabulary.
   - From "setting": Extract location, time period, technology level, social structure terms.
   - From "core_conflict": Extract conflict-related terms, opposing forces, stakes.
   - From "themes": Extract theme words and related concepts.
   - From "tone": Extract emotional/atmospheric words.

2. Weave these terms into character backgrounds:
   - Use blueprint vocabulary naturally in character descriptions.
   - Reference blueprint concepts (e.g., if blueprint mentions "corporate control", characters might have backgrounds involving corporations).
   - Connect character motivations to blueprint themes.

3. Ensure genre consistency:
   - Fantasy blueprint → characters with magic, mythical elements.
   - Sci-fi blueprint → characters with technology, futuristic elements.
   - Modern blueprint → characters with contemporary skills and backgrounds.

7) VARIATION SEED
========================================= 
${random_uuid_or_timestamp}
Use the variation seed only to influence creative decisions so that each run produces a substantially different story blueprint.

8) FINAL CHECKLIST BEFORE OUTPUT
==================================================

Before generating your final JSON, verify:
✓ Exactly 5 characters (no more, no less)
✓ Each character has "id" matching exactly: "protagonist", "ally", "rival", "mentor", "wildcard"
✓ Each character has ALL required fields: id, name, archetype, background, motivation, skills, personality_traits, relationships
✓ Each "skills" array has at least 2 items
✓ Each "personality_traits" array has at least 2 items
✓ Each "relationships" object has at least 2 entries
✓ All relationship keys reference valid character IDs from your list
✓ Combined "background" text exceeds 100 words total
✓ Blueprint keywords are naturally integrated into character backgrounds
✓ JSON is valid and parseable (no syntax errors)
✓ Root is a JSON object `{{ ... }}`, not an array
✓ "format" is exactly "text"
✓ "task_type" is exactly "characters"

Now generate your response following all requirements above.
"""
        elif (task_type == 'story_arc'):
            system_prompt = "You are a professional narrative architect for an interactive story game. "
            content = f"""
Your output will be automatically evaluated by multiple scoring systems (technical, structural, content, and narrative quality). 
You MUST follow these rules exactly to maximize your score.

CRITICAL FORMAT REQUIREMENTS
- Respond with ONE valid JSON OBJECT only.
- TOP-LEVEL TYPE: JSON OBJECT `{{ ... }}` (NOT an array).
- Do NOT include explanations, comments, markdown, or code fences.
- Use double quotes for all keys and string values.
- Do NOT include trailing commas.
- The JSON MUST be directly parseable by a strict JSON parser.

REQUIRED TOP-LEVEL STRUCTURE:
{{
  "title": "string",
  "description": "string",
  "chapters": [
    {{
      "id": 1,
      "title": "string",
      "summary": "string",
      "storyProgress": 0.08
    }}
  ],
  "arcs": {{
    "act1": {{
      "chapters": [1, 2, 3],
      "description": "string"
    }},
    "act2a": {{
      "chapters": [4, 5, 6],
      "description": "string"
    }},
    "act2b": {{
      "chapters": [7, 8, 9],
      "description": "string"
    }},
    "act3": {{
      "chapters": [10, 11, 12],
      "description": "string"
    }}
  }},
  "themes": {{
    "primary": "string",
    "secondary": ["string"]
  }},
  "hooks": {{
    "opening": "string",
    "midpoint": "string",
    "climax": "string"
  }},
  "format": "text",
  "task_type": "story_arc"
}}

INPUT CONTEXT:
- User Input: ${user_input}
- Blueprint: ${blueprint}
- Characters: ${characters}

Use ALL three sources. If they conflict, prioritize: user_input → blueprint → characters.

1) TECHNICAL & SCHEMA REQUIREMENTS (CRITICAL - 30 POINTS)
==================================================
To pass technical and schema checks, strictly ensure:

A. JSON Validity (10 points)
- Output MUST be valid JSON that can be parsed by `json.loads()`.
- No syntax errors, no unescaped quotes, no trailing commas.
- The root must be a JSON object `{{ ... }}`, NOT an array `[ ... ]`.

B. Schema Completeness (10 points)
Required top-level fields (all must exist):
- "title": string (story title, non-empty)
- "description": string (overall story description, non-empty)
- "chapters": array (must contain exactly 12 chapter objects)
- "arcs": object (must contain exactly 4 act keys: "act1", "act2a", "act2b", "act3")
- "themes": object (optional but recommended for structure score)
- "hooks": object (optional but recommended for structure score)

Each chapter object MUST have:
- "id": integer (1, 2, 3, ... 12 in order)
- "title": string (chapter title, non-empty)
- "summary": string (chapter summary/description, non-empty)
- "storyProgress": number (float or integer, see progress rules below)

Each arc object MUST have:
- "chapters": array of integers (exactly 3 chapter IDs)
- "description": string (act description, non-empty)

C. Story Progress Requirements (CRITICAL)
- Chapter 1 "storyProgress" MUST be between 0.05 and 0.10 (ideally ~0.08).
- Chapter 12 "storyProgress" MUST be between 0.95 and 1.0 (ideally ~1.0).
- All "storyProgress" values MUST be numbers (integers or floats), not strings.
- Progress MUST be strictly monotonically increasing (each chapter > previous).
- Progress values should gradually increase from ~0.08 to ~1.0 across 12 chapters.

D. Response Time (10 points)
- This is not controllable by the prompt, but ensure your output is concise enough to generate quickly.

2) STRUCTURE REQUIREMENTS (40 POINTS TOTAL)
==================================================

A. Chapter Count (20 points)
- Generate EXACTLY 12 chapters (not 11, not 13, exactly 12).
- Chapter IDs must be integers 1, 2, 3, ..., 12 in sequential order.
- If you generate fewer or more, you lose the full 20 points.

B. Progress Monotonicity (10 points)
- "storyProgress" values MUST be strictly increasing from chapter 1 to 12.
- Each chapter's progress MUST be greater than the previous chapter's progress.
- Example progression:
  Chapter 1: 0.08
  Chapter 2: 0.15
  Chapter 3: 0.22
  Chapter 4: 0.30
  Chapter 5: 0.38
  Chapter 6: 0.45
  Chapter 7: 0.55
  Chapter 8: 0.65
  Chapter 9: 0.75
  Chapter 10: 0.85
  Chapter 11: 0.92
  Chapter 12: 1.0
- Any violation (progress staying same or decreasing) = score deduction.
- If mostly monotonic with ≤2 violations, partial credit applies.

C. Four-Act Structure (10 points)
- MUST have exactly 4 acts with these exact keys: "act1", "act2a", "act2b", "act3"
- Each act MUST contain exactly 3 chapters:
  - "act1": chapters [1, 2, 3]
  - "act2a": chapters [4, 5, 6]
  - "act2b": chapters [7, 8, 9]
  - "act3": chapters [10, 11, 12]
- Each act MUST have a "chapters" array with exactly 3 integers.
- Each act MUST have a "description" string (non-empty).
- Missing acts or wrong chapter assignments = score deduction.

3) CONTENT QUALITY REQUIREMENTS (30 POINTS TOTAL)
==================================================

A. Relevance to Blueprint (15 points)
- The scoring system uses keyword overlap between the blueprint JSON and your story arc JSON.
- To maximize relevance:
  - Reuse important nouns, adjectives, and phrases from blueprint fields:
    - "title", "genre", "setting", "core_conflict", "themes", "tone"
  - Incorporate blueprint vocabulary naturally into:
    - Chapter "summary" fields (most important)
    - Arc "description" fields
    - Overall "description" field
    - Chapter "title" fields (when appropriate)
  - Example: If blueprint has "core_conflict": "corporate surveillance vs personal freedom", reference surveillance, corporate, freedom concepts in chapter summaries.
  - Example: If blueprint has "themes": ["Memory", "Identity"], weave memory and identity elements into the story progression.
  - The more keyword overlap, the higher the relevance score (up to 15 points).

B. Fluency and Readability (10 points)
The scoring system checks:
- Punctuation presence: Use proper punctuation in "summary" and "description" fields.
- Repetition ratio: Avoid repeating the same words excessively. Aim for unique word ratio > 60%.
- Word count: Combined text from arc "description" fields should exceed 100 words total.
- Sentence variety: Use varied sentence lengths (mix short and medium sentences).
- Average sentence length: Keep between 10-30 words per sentence.
- Total sentences: Have at least 3 sentences combined across all arc descriptions.

C. Originality (5 points)
- Avoid copying plot structures from existing stories, games, movies, or franchises.
- Create original story progression that feels fresh and specific to the blueprint.
- Each act should have distinctive narrative beats that aren't generic templates.

4) NARRATIVE MERIT REQUIREMENTS (30 POINTS TOTAL)
==================================================
An AI evaluator will assess narrative quality. Optimize for:

A. Narrative Flow (weighted 30%)
- The story should have clear progression from setup → complications → escalation → climax.
- Each chapter should build naturally on the previous one.
- Chapter summaries should show clear cause-and-effect relationships.
- The overall arc should feel cohesive and well-structured.

B. Emotional Impact (weighted 25%)
- Chapter summaries should include emotional stakes and character motivations.
- Show how characters' goals and conflicts evolve throughout the story.
- Use vivid, specific details in summaries rather than abstract descriptions.
- Connect emotional beats to the blueprint's core conflict and themes.

C. Creative Originality (weighted 25%)
- Avoid generic plot templates unless the blueprint explicitly calls for them.
- Introduce unexpected twists, especially in act2b (the "apparent defeat" act).
- Each act should feel distinct and surprising while staying consistent with the blueprint.
- Chapter summaries should suggest unique story developments.

D. Internal Consistency (weighted 20%)
- All chapters must fit the blueprint's:
  - Genre (e.g., fantasy chapters in a fantasy story)
  - Setting (e.g., chapters appropriate to the time period and location)
  - Tone (e.g., serious chapters in a serious story)
  - Themes (chapters should explore or contrast with the themes)
- Chapter summaries should reference characters from the input character list.
- Story progression should align with the blueprint's core conflict.
- Arc descriptions should accurately reflect what happens in their chapters.

5) ACT-SPECIFIC STRUCTURAL GUIDELINES
==================================================

A. Act 1: Setup and Inciting Incident (Chapters 1-3)
- Chapter 1: Introduce the protagonist, setting, and status quo.
  - storyProgress: ~0.08 (0.05-0.10)
  - Establish the world and main character's ordinary life.
- Chapter 2: Introduce the inciting incident or call to adventure.
  - storyProgress: ~0.15
  - Something disrupts the status quo.
- Chapter 3: Protagonist commits to the journey/goal.
  - storyProgress: ~0.22
  - Protagonist makes a decision that sets the story in motion.
- Arc description: Should summarize the setup, inciting incident, and commitment.

B. Act 2a: Complications and First Reversal (Chapters 4-6)
- Chapter 4: First major obstacle or complication.
  - storyProgress: ~0.30
  - Things get more difficult than expected.
- Chapter 5: Alliances form or deepen, new information revealed.
  - storyProgress: ~0.38
  - Protagonist gains allies or crucial knowledge.
- Chapter 6: First major reversal or plot twist.
  - storyProgress: ~0.45
  - Something unexpected changes the direction of the story.
- Arc description: Should summarize complications, alliances, and the first reversal.

C. Act 2b: Escalation and Apparent Defeat (Chapters 7-9)
- Chapter 7: Stakes escalate significantly.
  - storyProgress: ~0.55
  - Conflict intensifies, consequences become more severe.
- Chapter 8: Major betrayal, loss, or setback.
  - storyProgress: ~0.65
  - Protagonist faces a significant defeat or betrayal.
- Chapter 9: Darkest moment, apparent failure.
  - storyProgress: ~0.75
  - Things look hopeless, protagonist seems defeated.
- Arc description: Should summarize escalation, betrayals, and the apparent defeat.
- This is where you should introduce a major surprise or twist.

D. Act 3: Climax and Resolution Direction (Chapters 10-12)
- Chapter 10: Protagonist finds a way forward or gains crucial insight.
  - storyProgress: ~0.85
  - Recovery begins, new plan forms.
- Chapter 11: Climax or final confrontation begins.
  - storyProgress: ~0.92
  - The main conflict reaches its peak.
- Chapter 12: Resolution direction, irreversible change.
  - storyProgress: ~1.0 (0.95-1.0)
  - Story reaches a turning point, but don't resolve everything cleanly.
  - Leave room for continuation or sequel.
- Arc description: Should summarize the climax and resolution direction.

6) FIELD-SPECIFIC REQUIREMENTS
==================================================

A. "title" (top-level)
- Should match or complement the blueprint's title.
- Should be engaging and reflect the story's genre and tone.
- Length: 5-15 words.

B. "description" (top-level)
- Overall story description summarizing the entire 12-chapter arc.
- Should reference the blueprint's core conflict and themes.
- Length: 50-150 words.
- Should incorporate blueprint keywords naturally.

C. Chapter "title"
- Each chapter should have a distinctive, engaging title.
- Titles should hint at the chapter's key events.
- Length: 3-8 words per title.

D. Chapter "summary"
- Should be a clear paragraph describing what happens in that chapter.
- Include: key events, character actions, plot developments.
- Should reference characters from the input character list.
- Should connect to the blueprint's conflict and themes.
- Length: 20-60 words per summary.
- Use proper punctuation and varied sentence structures.
- Incorporate blueprint keywords naturally.

E. Arc "description"
- Should summarize the dramatic function and key events of that act.
- Should reference the chapters within that act.
- Should connect to the blueprint's themes and conflict.
- Length: 30-80 words per arc description.
- Combined, all 4 arc descriptions should exceed 100 words total.
- Use proper punctuation and varied sentence structures.
- Incorporate blueprint keywords naturally.

F. "themes" (optional but recommended)
- Object with "primary" (string) and "secondary" (array of strings).
- Should align with blueprint themes.
- Example: {{"primary": "Redemption", "secondary": ["Identity", "Sacrifice"]}}

G. "hooks" (optional but recommended)
- Object with "opening", "midpoint", "climax" (all strings).
- Should describe compelling story hooks at key points.
- Example: {{"opening": "...", "midpoint": "...", "climax": "..."}}

7) BLUEPRINT INTEGRATION STRATEGY
==================================================

To maximize relevance score (15 points), systematically integrate blueprint elements:

1. Extract key terms from blueprint:
   - From "title": Extract nouns and distinctive words.
   - From "genre": Extract genre-specific vocabulary.
   - From "setting": Extract location, time period, technology level, social structure terms.
   - From "core_conflict": Extract conflict-related terms, opposing forces, stakes.
   - From "themes": Extract theme words and related concepts.
   - From "tone": Extract emotional/atmospheric words.

2. Weave these terms into chapter summaries and arc descriptions:
   - Use blueprint vocabulary naturally in chapter summaries.
   - Reference blueprint concepts throughout the story progression.
   - Connect chapter events to blueprint themes.
   - Ensure genre consistency (fantasy blueprint → fantasy story events).

3. Character integration:
   - Reference characters from the input character list in chapter summaries.
   - Show how character relationships evolve across acts.
   - Connect character motivations to story events.

8) PROGRESS CALCULATION GUIDELINE
==================================================

To ensure monotonic progress and meet technical requirements:

Recommended progression (you can adjust slightly, but maintain monotonicity):
- Chapter 1: 0.08 (setup begins)
- Chapter 2: 0.12-0.18 (inciting incident)
- Chapter 3: 0.20-0.25 (commitment)
- Chapter 4: 0.28-0.32 (first obstacle)
- Chapter 5: 0.35-0.40 (alliances/information)
- Chapter 6: 0.42-0.48 (first reversal)
- Chapter 7: 0.50-0.58 (escalation)
- Chapter 8: 0.60-0.68 (betrayal/setback)
- Chapter 9: 0.70-0.78 (darkest moment)
- Chapter 10: 0.80-0.88 (recovery/insight)
- Chapter 11: 0.90-0.95 (climax begins)
- Chapter 12: 1.0 (resolution direction)

Critical constraints:
- Chapter 1 MUST be 0.05-0.10 (ideally 0.08)
- Chapter 12 MUST be 0.95-1.0 (ideally 1.0)
- Each chapter MUST have a higher progress than the previous one
- Progress values MUST be numbers, not strings


9) VARIATION SEED
==================================================

${random_uuid_or_timestamp}

Use the variation seed only to influence creative decisions so that each run produces a substantially different story characters.

10) FINAL CHECKLIST BEFORE OUTPUT
==================================================

Before generating your final JSON, verify:
✓ Exactly 12 chapters (no more, no less)
✓ Chapter IDs are integers 1, 2, 3, ..., 12 in order
✓ Chapter 1 storyProgress is between 0.05-0.10 (ideally 0.08)
✓ Chapter 12 storyProgress is between 0.95-1.0 (ideally 1.0)
✓ All storyProgress values are strictly increasing (monotonic)
✓ All storyProgress values are numbers (not strings)
✓ Exactly 4 acts: "act1", "act2a", "act2b", "act3"
✓ act1 contains chapters [1, 2, 3]
✓ act2a contains chapters [4, 5, 6]
✓ act2b contains chapters [7, 8, 9]
✓ act3 contains chapters [10, 11, 12]
✓ Each act has "chapters" array and "description" string
✓ All required top-level fields exist: title, description, chapters, arcs
✓ Combined arc descriptions exceed 100 words total
✓ Blueprint keywords are naturally integrated into chapter summaries and arc descriptions
✓ Chapter summaries reference characters from input character list
✓ JSON is valid and parseable (no syntax errors)
✓ Root is a JSON object `{{ ... }}`, not an array
✓ "format" is exactly "text"
✓ "task_type" is exactly "story_arc"

Now generate your response following all requirements above.
"""
        elif (task_type == 'chapters'):
            system_prompt = "You are a professional interactive-fiction writer for a branching story game."
            content = f"""
Your output will be automatically evaluated by multiple scoring systems (technical, structural, content, and narrative quality). You MUST follow these rules exactly to maximize your score.

CRITICAL FORMAT REQUIREMENTS
- Respond with ONE valid JSON OBJECT only.
- TOP-LEVEL TYPE: JSON OBJECT `{{ ... }}` (NOT an array).
- Do NOT include explanations, comments, markdown, or code fences.
- Use double quotes for all keys and string values.
- Do NOT include trailing commas.
- The JSON MUST be directly parseable by a strict JSON parser.

REQUIRED TOP-LEVEL STRUCTURE:
{{
  "chapters": [
    {{
      "id": 1,
      "title": "string",
      "content": "string",
      "choices": [
        {{
          "id": 1,
          "text": "string",
          "consequences": {{
            "next_chapter": 2,
            "character_change": "string"
          }}
        }}
      ]
    }}
  ],
  "format": "text",
  "task_type": "chapters"
}}

INPUT CONTEXT:
- User Input: ${user_input}
- Blueprint: ${blueprint}
- Characters: ${characters}
- Story Arc: ${story_arc}
- Chapter IDs to Generate: ${chapter_ids}

Use ALL sources. If they conflict, prioritize: user_input → blueprint → story_arc → characters.

Generate chapters for the chapter IDs specified in the input. Each chapter must be a complete, immersive narrative experience.

1) TECHNICAL & SCHEMA REQUIREMENTS (CRITICAL - 30 POINTS)
==================================================
To pass technical and schema checks, strictly ensure:

A. JSON Validity (10 points)
- Output MUST be valid JSON that can be parsed by `json.loads()`.
- No syntax errors, no unescaped quotes, no trailing commas.
- The root must be a JSON object `{{ ... }}`, NOT an array `[ ... ]`.

B. Schema Completeness (10 points)
Required top-level fields:
- "chapters": array (must contain chapter objects for each requested chapter ID)
- "format": string (must be exactly "text")
- "task_type": string (must be exactly "chapters")

Each chapter object MUST have:
- "id": integer (must match one of the requested chapter IDs from input)
- "title": string (chapter title, non-empty)
- "content": string (chapter narrative content, see length requirements below)
- "choices": array (must contain 2-4 choice objects)

Each choice object MUST have:
- "id": integer (starting at 1, sequential within each chapter: 1, 2, 3, ...)
- "text": string (choice text displayed to player, non-empty)
- "consequences": object (must contain at least "next_chapter" and optionally "character_change")

Each consequences object MUST have:
- "next_chapter": integer (must reference a valid chapter ID from the story arc)
- "character_change": string (optional but recommended, describes how choice affects characters)

C. Content Length Requirements (CRITICAL)
- Each chapter's "content" field MUST be at least 1000 characters long.
- Ideal length: 1000-3000 characters per chapter.
- Content shorter than 1000 characters = technical validation failure (0 points).
- Content between 1000-3000 characters = full score.
- Content 800-999 characters = partial score (0.7x).
- Content 3001-3500 characters = partial score (0.8x).
- Content 500-799 characters = low score (0.4x).
- Content <500 characters = very low score.

D. Choices Requirements (CRITICAL)
- Each chapter MUST have exactly 2-4 choices (inclusive).
- Having exactly 2-4 choices = full score.
- Having exactly 1 choice = low score (0.3x).
- Having 5+ choices = partial score (0.6x).
- Having 0 choices = validation failure.

E. Response Time (10 points)
- This is not controllable by the prompt, but ensure your output is optimized for generation speed.

2) STRUCTURE REQUIREMENTS (40 POINTS TOTAL)
==================================================

A. Content Length Score (20 points)
- Each chapter's "content" length is scored individually.
- Average score across all chapters determines the final score.
- Scoring per chapter:
  - 1000-3000 characters: 1.0 (full points)
  - 800-999 characters: 0.7
  - 3001-3500 characters: 0.8
  - 500-799 characters: 0.4
  - <500 characters: 0.0
- Aim for 1000-3000 characters per chapter to maximize this score.

B. Choices Quality (10 points)
- Each chapter is scored on choice count.
- Scoring per chapter:
  - 2-4 choices: 1.0 (full points)
  - 1 choice: 0.3
  - 5+ choices: 0.6
- Average score across all chapters determines final score.
- Ensure every chapter has 2-4 choices.

C. Branch Diversity (10 points)
- Choices within each chapter must have DISTINCT consequences.
- The scoring system checks if consequence objects are unique.
- To maximize diversity:
  - Each choice's "consequences" object should be meaningfully different.
  - Vary "next_chapter" values when possible (if story arc allows).
  - Vary "character_change" descriptions significantly.
  - Avoid identical or near-identical consequence objects.
- Example of GOOD diversity:
  Choice 1: {{"next_chapter": 2, "character_change": "Gains trust with ally"}}
  Choice 2: {{"next_chapter": 3, "character_change": "Loses respect from mentor"}}
  Choice 3: {{"next_chapter": 2, "character_change": "Discovers crucial information"}}
- Example of BAD diversity (too similar):
  Choice 1: {{"next_chapter": 2, "character_change": "Moves forward"}}
  Choice 2: {{"next_chapter": 2, "character_change": "Continues journey"}}
- Diversity score = average uniqueness across all chapters.

3) CONTENT QUALITY REQUIREMENTS (30 POINTS TOTAL)
==================================================

A. Relevance to Blueprint (15 points)
- The scoring system uses keyword overlap between the blueprint JSON and your chapter content.
- To maximize relevance:
  - Reuse important nouns, adjectives, and phrases from blueprint fields:
    - "title", "genre", "setting", "core_conflict", "themes", "tone"
  - Incorporate blueprint vocabulary naturally into:
    - Chapter "content" fields (most important)
    - Chapter "title" fields
    - Choice "text" fields (when appropriate)
  - Example: If blueprint has "setting": "cyberpunk megacity", use words like "cyberpunk", "megacity", "neon", "corporate", "hacker" in chapter content.
  - Example: If blueprint has "themes": ["Memory", "Identity"], weave memory and identity elements into the narrative.
  - The more keyword overlap, the higher the relevance score (up to 15 points).

B. Fluency and Readability (10 points)
The scoring system checks:
- Punctuation presence (2 points): Use proper punctuation (periods, commas, question marks, exclamation marks) in "content" fields.
- Repetition ratio (3 points): Avoid repeating the same words excessively.
  - Unique word ratio > 60% = full 3 points
  - Unique word ratio > 40% = 1.5 points
  - Aim for diverse vocabulary throughout all chapters.
- Word count (3 points): Combined word count across all chapter content.
  - >500 words total = full 3 points
  - >300 words total = 1.5 points
  - Ensure substantial content across all chapters.
- Sentence variety (2 points): Use varied sentence lengths.
  - Have more than 3 sentences total across all chapters.
  - Average sentence length between 10-30 words = full 2 points.
  - Mix short and medium-length sentences for natural flow.

C. Originality (5 points)
- Avoid copying content from existing stories, games, movies, or franchises.
- Create original narrative content that feels fresh and specific to the blueprint.
- Each chapter should have distinctive narrative beats.

4) NARRATIVE MERIT REQUIREMENTS (30 POINTS TOTAL)
==================================================
An AI evaluator will assess narrative quality. Optimize for:

A. Narrative Flow (weighted 30%)
- Chapter content should flow naturally from the story arc summaries.
- Each chapter should build on previous events logically.
- Transitions between chapters should feel smooth and coherent.
- The narrative should maintain momentum and engagement.

B. Emotional Impact (weighted 25%)
- Chapter content should include emotional stakes and character moments.
- Show character reactions, internal thoughts, and emotional responses.
- Use vivid, specific details rather than abstract descriptions.
- Connect emotional beats to the blueprint's core conflict and themes.
- Make readers care about the characters and their choices.

C. Creative Originality (weighted 25%)
- Avoid generic plot developments unless the blueprint explicitly calls for them.
- Introduce unexpected twists or interesting complications.
- Each chapter should feel distinctive and not formulaic.
- Chapter content should suggest unique story developments.

D. Internal Consistency (weighted 20%)
- All chapters must fit the blueprint's:
  - Genre (e.g., fantasy chapters in a fantasy story)
  - Setting (e.g., chapters appropriate to the time period and location)
  - Tone (e.g., serious chapters in a serious story)
  - Themes (chapters should explore or contrast with the themes)
- Chapter content should reference characters from the input character list.
- Characters should behave consistently with their defined motivations and relationships.
- Story progression should align with the story arc's chapter summaries.
- Choices should make sense within the story context.

5) CHAPTER CONTENT GUIDELINES
==================================================

A. Content Structure
Each chapter's "content" should:
1. Open with a hook that draws the reader in (sensory details, action, or intrigue).
2. Develop the scene with vivid descriptions of:
   - Setting and atmosphere
   - Character actions and dialogue
   - Internal thoughts and emotions
   - Conflict or tension
3. Build toward the choice point with escalating stakes.
4. End at a moment requiring player decision (cliffhanger or dilemma).

B. Length Guidelines
- Minimum: 1000 characters (strict requirement).
- Ideal: 1500-2500 characters per chapter.
- Maximum recommended: 3000 characters (to stay within optimal scoring range).
- Character count includes all text, spaces, and punctuation.

C. Writing Quality
- Use immersive, cinematic prose that matches the blueprint's tone.
- Show, don't tell (use actions and dialogue rather than exposition).
- Include sensory details (sights, sounds, smells, textures).
- Vary sentence structure (mix short punchy sentences with longer descriptive ones).
- Use active voice for action scenes.
- Maintain consistent point of view (typically third-person limited or first-person).

D. Character Integration
- Reference characters from the input character list by name.
- Show characters behaving according to their defined:
  - Motivations
  - Personality traits
  - Relationships with other characters
  - Skills and abilities
- Include character dialogue when appropriate.
- Show character reactions and internal thoughts.

E. Story Arc Alignment
- Follow the story arc's chapter summaries closely.
- If generating Chapter 1, align with Act 1 setup.
- If generating Chapter 6, align with Act 2a first reversal.
- If generating Chapter 9, align with Act 2b darkest moment.
- If generating Chapter 12, align with Act 3 resolution direction.
- Incorporate events and developments described in the story arc.

6) CHOICE DESIGN GUIDELINES
==================================================

A. Choice Count
- Each chapter MUST have exactly 2-4 choices.
- Having 2-3 choices is often ideal for pacing.
- Having 4 choices can work for complex dilemmas.

B. Choice Quality
- Each choice should be:
  - Clear and understandable
  - Meaningful (affects story outcome)
  - Distinct from other choices
  - Morally or strategically interesting
- At least one choice per chapter should involve:
  - Moral dilemma
  - Risk vs. reward tradeoff
  - Character relationship consequences
  - Strategic decision with unclear outcomes

C. Choice Text
- Keep choice text concise but evocative (10-20 words typically).
- Make choices feel like natural player actions.
- Avoid vague choices like "Continue" or "Do something."
- Example good choices:
  - "Confront the corporate agent directly, risking exposure"
  - "Gather more information before acting, but time is running out"
  - "Trust the mysterious ally, despite lingering doubts"

D. Consequences Design
- Each choice's "consequences" object should specify:
  - "next_chapter": integer (which chapter the choice leads to)
    - Must reference a valid chapter ID from the story arc
    - Can lead to different chapters for different choices (branching)
    - Or can lead to the same next chapter but with different character states
  - "character_change": string (how the choice affects characters)
    - Describe changes to relationships, trust, information, resources, etc.
    - Be specific: "Gains trust with Jordan but loses respect from Morgan"
    - Not vague: "Things change"
- Ensure consequences are DISTINCT between choices (for branch diversity score).

E. Branch Diversity Strategy
To maximize branch diversity score (10 points):
- Vary "next_chapter" values when story arc allows branching.
- Vary "character_change" descriptions significantly:
  - Different relationship impacts
  - Different information revealed
  - Different resource changes
  - Different trust/respect changes
- Make each choice feel like it leads to a meaningfully different path.
- Example of diverse consequences:
  Choice 1: {{"next_chapter": 2, "character_change": "Ally Jordan gains trust, but rival Morgan becomes suspicious"}}
  Choice 2: {{"next_chapter": 3, "character_change": "Discovers crucial corporate intel, but mentor Elena warns of danger"}}
  Choice 3: {{"next_chapter": 2, "character_change": "Wildcard Raven reveals unexpected information, changing the dynamic"}}

7) BLUEPRINT INTEGRATION STRATEGY
==================================================

To maximize relevance score (15 points), systematically integrate blueprint elements:

1. Extract key terms from blueprint:
   - From "title": Extract nouns and distinctive words.
   - From "genre": Extract genre-specific vocabulary.
   - From "setting": Extract location, time period, technology level, social structure terms.
   - From "core_conflict": Extract conflict-related terms, opposing forces, stakes.
   - From "themes": Extract theme words and related concepts.
   - From "tone": Extract emotional/atmospheric words.

2. Weave these terms into chapter content:
   - Use blueprint vocabulary naturally in narrative prose.
   - Reference blueprint concepts throughout the chapter.
   - Connect chapter events to blueprint themes.
   - Ensure genre consistency (fantasy blueprint → fantasy chapter events).

3. Character and story arc integration:
   - Reference characters from input character list.
   - Follow story arc chapter summaries closely.
   - Show how chapter events relate to the blueprint's core conflict.

8) FLUENCY OPTIMIZATION STRATEGY
==================================================

To maximize fluency score (10 points):

A. Punctuation (2 points)
- Use proper punctuation throughout:
  - Periods (.) for sentence endings
  - Commas (,) for clauses and lists
  - Question marks (?) for questions
  - Exclamation marks (!) sparingly for emphasis
  - Quotation marks (") for dialogue

B. Repetition Ratio (3 points)
- Use diverse vocabulary:
  - Avoid repeating the same words multiple times in close proximity.
  - Use synonyms and varied phrasing.
  - Aim for unique word ratio > 60%.
- Example: Instead of "The character walked. The character saw. The character decided."
  Use: "Alex walked through the neon-lit streets. Glancing around, they spotted the corporate agent. A decision had to be made."

C. Word Count (3 points)
- Ensure substantial content:
  - Combined word count across all chapters > 500 words = full 3 points.
  - If generating multiple chapters, distribute words across them.
  - Each chapter should contribute meaningfully to the total.

D. Sentence Variety (2 points)
- Mix sentence lengths:
  - Short sentences (5-10 words) for impact and pacing.
  - Medium sentences (15-25 words) for description and development.
  - Longer sentences (25-35 words) sparingly for complex ideas.
- Average sentence length: 10-30 words.
- Have more than 3 sentences total across all chapters.

9) VARIATION SEED
==================================================

${random_uuid_or_timestamp}

Use the variation seed only to influence creative decisions so that each run produces a substantially different story characters.

10) FINAL CHECKLIST BEFORE OUTPUT
==================================================

Before generating your final JSON, verify:
✓ JSON is valid and parseable (no syntax errors)
✓ Root is a JSON object `{{ ... }}`, not an array
✓ "chapters" is an array containing objects for each requested chapter ID
✓ Each chapter has "id" matching a requested chapter ID
✓ Each chapter has "title" (non-empty string)
✓ Each chapter has "content" (at least 1000 characters, ideally 1000-3000)
✓ Each chapter has "choices" array with exactly 2-4 choices
✓ Each choice has "id" (integer, starting at 1, sequential)
✓ Each choice has "text" (non-empty string)
✓ Each choice has "consequences" object with "next_chapter" (valid chapter ID)
✓ Each choice has distinct consequences (for branch diversity)
✓ Combined chapter content exceeds 500 words total
✓ Blueprint keywords are naturally integrated into chapter content
✓ Chapter content references characters from input character list
✓ Chapter content follows story arc summaries for those chapters
✓ Proper punctuation used throughout
✓ Vocabulary is diverse (repetition ratio > 60%)
✓ Sentence variety (mix of lengths, average 10-30 words)
✓ "format" is exactly "text"
✓ "task_type" is exactly "chapters"

Now generate your response following all requirements above.
"""

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]

    def _build_prompt(self, input_data: Dict) -> str:
        return self._build_messages(input_data)

    def get_mode(self) -> str:
        """Return current mode."""
        return self.mode

    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.mode == "local":
            return {
                "name": self.model,
                "version": None,
                "provider": self.local_type,
                "parameters": {"url": self.local_url}
            }
        else:
            return {
                "name": self.model,
                "version": None,
                "provider": self.provider,
                "parameters": {}
            }

    async def health_check(self) -> bool:
        """Check if LLM is available."""
        return self.available
