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
        random_uuid_or_timestamp = int(time.time())

        if (task_type == 'blueprint'):
            system_prompt = "You are a professional narrative designer for an interactive story game."
            content = f"""Given the following user_input, generate a creative, entertaining, and entirely new story blueprint.
            You MUST return a single valid JSON object in exactly the following format and with ALL required fields:
            {{
                "title": "string",
                "genre": "string",
                "setting": "string",
                "core_conflict": "string",
                "themes": ["string"],
                "tone": "string",
                "target_audience": "string",
                "generated_text": "string",
                "format": "text",
                "task_type": "blueprint"
            }}

            ────────────────────────────
            ### STRICT RULES

            - Output ONLY valid JSON.
            - Do NOT include explanations, markdown, or additional text.
            - All fields must be present.
            - Infer values from the user_input.
            - "setting" must be between **50 and 300 characters**.
            - "core_conflict" must be between **20 and 200 characters**.
            - "themes" must contain **2–5 items**.
            - "generated_text" must be a full narrative blueprint describing the world, premise, stakes, and story direction.
            - "format" must be exactly: "text".
            - "task_type" must be exactly: "blueprint".

            ────────────────────────────
            ### NOVELTY & VARIATION REQUIREMENTS

            - Each response MUST describe a completely new story world, premise, and conflict.
            - Do NOT reuse plots, factions, locations, power systems, or narrative structures from previous responses.
            - Avoid generic or overused tropes unless explicitly requested.
            - Randomize at least THREE of the following every time:
            - Time period
            - Cultural inspiration
            - Central threat or mystery
            - Technology or magic system
            - Social structure
            - Setting scale

            - If multiple interpretations of user_input exist, select an unusual but plausible angle.
            - Before writing the final output, internally consider several story concepts and choose the most distinctive one.

            - Have the word repetitation ratio to less than 40% from settings and core_conflict
            - Have the word count more than 100 from settings and core_conflict
            - Have some duplicated words between user_input and title + genre + setting

            ────────────────────────────
            ### VARIATION SEED

            ${random_uuid_or_timestamp}

            Use the variation seed only to influence creative decisions so that each run produces a substantially different story.

            ────────────────────────────
            ### USER INPUT

            ${user_input}
            """
        elif (task_type == 'characters'):
            system_prompt = "You are a professional character designer for an interactive story game."
            content = f"""
                You will be given:
                1) a story blueprint
                2) additional user_input that may refine or override details

                Use BOTH sources when designing characters. If they conflict, prefer the user_input.
                You MUST return a single valid JSON object in exactly the following format:

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
                    "generated_text": "string",
                    "format": "text",
                    "task_type": "characters"
                }}

                ────────────────────────────
                ### STRICT RULES

                - Output ONLY valid JSON.
                - Do NOT include explanations, markdown, or additional text.
                - Generate **exactly 5 characters**.
                - Every character must include ALL required fields.
                - Each character must define **at least 2 relationships** referencing other character IDs in the same list.
                - Relationship references must be bidirectionally consistent.
                - All IDs must be unique and stable strings (e.g., "c1", "c2", etc.).
                - "skills" must contain at least 2 items.
                - "personality_traits" must contain at least 2 items.
                - "format" must be exactly: "text".
                - "task_type" must be exactly: "characters".

                ────────────────────────────
                ### CREATIVE & NOVELTY REQUIREMENTS

                - Characters must reflect the blueprint’s setting, themes, tone, and conflict.
                - Avoid generic archetypes unless explicitly requested.
                - No two characters may share the same archetype.
                - Each character must differ in worldview, social role, and moral alignment.
                - At least one relationship must involve rivalry, secrecy, or ideological conflict.
                
                - Have the word repetitation ratio to less than 40% from all characters' description
                - Have the word count more than 100 from all characters' description
                - Keep the total number of sentences more than 3 and keep the average sentence length to **10-30** words
                
                - Have some duplicated words betwee blueprint and characters' description

                ────────────────────────────
                ### GENERATED_TEXT REQUIREMENTS

                - "generated_text" should summarize how the five characters connect and collectively propel the narrative.

                ────────────────────────────
                ### INPUT BLUEPRINT

                {{blueprint}}

                ────────────────────────────
                ### USER INPUT

                {{user_input}}
            """        
        elif (task_type == 'story_arc'):
            system_prompt = "You are a professional narrative architect for an interactive story game."
            content = f"""
                You will be given:
                1) user_input
                2) a story blueprint
                3) a list of characters

                Use ALL three sources to design a coherent 12-chapter story outline with a four-act dramatic structure.
                If any sources conflict, prioritize the user_input, then the blueprint, then the characters.
                You MUST return a single valid JSON object in exactly the following format:

                {{
                    "title": "string",
                    "chapters": [
                        {{
                            "id": 1,
                            "title": "string",
                            "summary": "string",
                            "storyProgress": 10
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
                    "generated_text": "string",
                    "format": "text",
                    "task_type": "story_arc"
                }}

                ────────────────────────────
                ### STRICT RULES

                - Output ONLY valid JSON.
                - Do NOT include explanations, markdown, or additional text.
                - Generate **exactly 12 chapters**.
                - Chapter IDs must be integers 1–12 in order.
                - "storyProgress" must strictly increase from chapter 1 to 12 and stay within 0–100.
                - Each chapter must escalate stakes or consequences.
                - The four arcs MUST exist exactly as:
                - act1 → chapters [1,2,3]
                - act2a → chapters [4,5,6]
                - act2b → chapters [7,8,9]
                - act3 → chapters [10,11,12]
                - Arc descriptions must reflect the actual events in their chapters.
                - "format" must be exactly: "text".
                - "task_type" must be exactly: "story_arc".

                ────────────────────────────
                ### STRUCTURAL REQUIREMENTS

                - act1 = setup, inciting incident, commitment.
                - act2a = complications, alliances, first major reversal.
                - act2b = escalation, betrayals, apparent defeat.
                - act3 = climax, resolution direction, irreversible change.
                - Do NOT resolve every conflict cleanly; leave room for continuation.

                ────────────────────────────
                ### CREATIVE & NOVELTY REQUIREMENTS

                - Avoid generic plot templates unless user_input explicitly demands them.
                - Introduce at least one major surprise in act2b.
                - Consequences must permanently change relationships or the world.
                - Each act must feel tonally distinct while staying consistent with the blueprint.
                
                - Have the word repetitation ratio to less than 40% from all chapters' description
                - Have the word count more than 100 from all chapters' description
                - Keep the total number of sentences more than 3 and keep the average sentence length to **10-30** words
                
                - Have some duplicated words between blueprint and description

                ────────────────────────────
                ### GENERATED_TEXT REQUIREMENTS

                - "generated_text" should summarize the full dramatic trajectory across all four acts.

                ────────────────────────────
                ### USER INPUT

                {{user_input}}

                ────────────────────────────
                ### INPUT BLUEPRINT

                {{blueprint}}

                ────────────────────────────
                ### INPUT CHARACTERS

                {{characters}}

            """
        elif (task_type == 'chapters'):
            system_prompt = "You are a professional interactive-fiction writer for a branching story game."
            content = f"""
                You will be given:
                1) user_input
                2) a story blueprint
                3) a list of characters
                4) a story arc containing chapter outlines

                Use ALL of these sources to write full narrative chapters with player choices.

                If any sources conflict, prioritize them in this order:
                user_input → blueprint → story_arc → characters.

                You MUST return a single valid JSON object in exactly the following format:

                {{
                    "chapters": [
                        {{
                            "id": 1,
                            "content": "string",
                            "title": "string",
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
                    "generated_text": "string",
                    "format": "text",
                    "task_type": "chapters"
                }}

                ────────────────────────────
                ### STRICT RULES

                - Output ONLY valid JSON.
                - Do NOT include explanations, markdown, or additional text.
                - Each chapter’s "content" must be **1000–3000 characters**.
                - Each chapter must contain **2–4 choices**.
                - Each choice must have **distinct consequences**.
                - Each "next_chapter" must reference a valid chapter ID in the story arc.
                - Choice IDs must be integers starting at 1 within each chapter.
                - "format" must be exactly: "text".
                - "task_type" must be exactly: "chapters".

                ────────────────────────────
                ### NARRATIVE REQUIREMENTS

                - Chapter prose must be immersive, cinematic, and consistent with tone and genre.
                - Follow the story_arc summaries closely.
                - Characters must behave consistently with their motivations and relationships.
                - Each chapter must end at a moment requiring player decision.
                - Consequences must alter:
                - alliances
                - information revealed
                - injuries or resources
                - trust levels
                - future obstacles

                ────────────────────────────
                ### CREATIVE & NOVELTY REQUIREMENTS

                - Avoid repetitive chapter structures.
                - At least one choice per chapter should be morally difficult or risky.
                - Different choices must lead to meaningfully different narrative trajectories.

                - Have the word repetitation ratio to less than 40% from content
                - Have the word count more than 500 from content
                - Keep the total number of sentences more than 3 and keep the average sentence length to **10-30** words
                - Have some duplicated words between blueprint and content

                ────────────────────────────
                ### GENERATED_TEXT REQUIREMENTS

                - "generated_text" should summarize how the chapters work together as an interactive experience and how player agency shapes outcomes.

                ────────────────────────────
                ### USER INPUT

                {{user_input}}

                ────────────────────────────
                ### INPUT BLUEPRINT

                {{blueprint}}

                ────────────────────────────
                ### INPUT CHARACTERS

                {{characters}}

                ────────────────────────────
                ### INPUT STORY ARC

                {{story_arc}}
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
