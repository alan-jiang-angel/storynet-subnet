import bittensor as bt
import asyncio
import os
import json
import random


from dotenv import load_dotenv
from template.protocol import (
    StoryGenerationSynapse, 
    create_blueprint_synapse, 
    create_characters_synapse, 
    create_story_arc_synapse, 
    create_chapters_synapse
)
from generators.loader import GeneratorLoader

import json
from typing import Any, Dict, Optional

from scoring.technical import calculate_technical_score
from scoring.structure import calculate_structure_score
from scoring.content import calculate_content_score
from scoring.narrative import calculate_narrative_score  # optional


load_dotenv()

class TestRunner:
    def __init__(self):
        self.generator = GeneratorLoader()
        
    async def generate_response(self, synapse: StoryGenerationSynapse):
        input_data = {
            "user_input": synapse.user_input,
            "blueprint": synapse.blueprint,
            "characters": synapse.characters,
            "story_arc": synapse.story_arc,
            "chapter_ids": synapse.chapter_ids,
            "task_type": synapse.task_type  # Pass task type to generator
        }
        result = await self.generator.generate(input_data)

        # print("=================response=================")
        # print(result)

        return result
    
    def test_scoring(self, response_json):
        pass


DEFAULT_REQUIRED_FIELDS = {
    "blueprint": ["title", "genre", "setting", "core_conflict", "themes", "tone", "target_audience"],
    "characters": ["characters"],
    "story_arc": ["chapters"],
    "chapters": ["chapters"],
}

def evaluate_response(
    response: Any,
    generation_time: float,
    task_type: str,
    context: Optional[Dict] = None,
    history: Optional[list] = None,
    include_narrative: bool = False
) -> Dict[str, Any]:
    context = context or {}
    history = history or []

    # Normalize/parse response
    parsed = None
    if isinstance(response, dict):
        parsed = response
        response_json = json.dumps(response, ensure_ascii=False)
    else:
        response_json = str(response or "")
        try:
            parsed_candidate = json.loads(response_json)
            if isinstance(parsed_candidate, dict):
                parsed = parsed_candidate
        except Exception:
            parsed = None

    required_fields = DEFAULT_REQUIRED_FIELDS.get(task_type, [])

    # Technical (0-30)
    tech_score, tech_break = calculate_technical_score(response_json, generation_time, task_type, required_fields)

    # Structure (0-40)
    if parsed is None:
        struct_score, struct_break = 0.0, {}
    else:
        struct_score, struct_break = calculate_structure_score(parsed, task_type)

    # Content (0-30)
    if parsed is None:
        pseudo = {"content": response_json}
        content_score, content_break = calculate_content_score(pseudo, context, task_type, history)
    else:
        content_score, content_break = calculate_content_score(parsed, context, task_type, history)

    # Optional narrative (0-30) - uses AI evaluator in scoring.narrative
    narrative_score = None
    narrative_break = None
    if include_narrative and parsed is not None:
        narrative_score, narrative_break = calculate_narrative_score(parsed, context, task_type)

    # Aggregate
    base_score = tech_score + struct_score + content_score
    base_score = max(0.0, min(base_score, 100.0))

    report = {
        "base_score_3part": base_score,                   # technical+structure+content (0-100)
        "components": {
            "technical": {"score": tech_score, "breakdown": tech_break},
            "structure": {"score": struct_score, "breakdown": struct_break},
            "content": {"score": content_score, "breakdown": content_break},
        },
        "normalized": {
            "technical": tech_score / 30.0,
            "structure": struct_score / 40.0,
            "content": content_score / 30.0,
            "overall": base_score / 100.0
        },
        "parsed": parsed,
        # "raw": response_json
    }

    if narrative_score is not None:
        report["narrative"] = {"score": narrative_score, "breakdown": narrative_break}
        report["base_score_with_narrative"] = max(0.0, min(base_score + narrative_score, 130.0))

    return report

async def main():
    tester = TestRunner()

    path = "miner_input-02-02.jsonl"
    selected = None

    with open(path, "r", encoding="utf-8") as f:
        ## Read Random Line
        # for i, line in enumerate(f, 1):
        #     # Reservoir sampling
        #     if random.randrange(i) == 0:
        #         selected = line

        ## Read a speicific line
        # line_number = 31 # blueprint
        line_number = 9 # characters
        # line_number = 7 # story_arc
        # line_number = 16 # chapters
        
        for i, line in enumerate(f):
            if i == line_number:
                obj = json.loads(line)
                break

    # obj = json.loads(selected)
    data = obj.get("data")
    
    task_type = data.get("task_type")
    print(f"Running test for task type: {task_type}")
    
    if task_type == "blueprint":
        incoming_synapse = create_blueprint_synapse(data.get("user_input", ""))
    elif task_type == "characters":
        incoming_synapse = create_characters_synapse(data.get("blueprint", {}), data.get("user_input", ""))
    elif task_type == "story_arc":
        incoming_synapse = create_story_arc_synapse(data.get("blueprint", {}), data.get("characters", []), data.get("user_input", ""))
    elif task_type == "chapters":
        incoming_synapse = create_chapters_synapse(data.get("blueprint", {}), data.get("characters", []), data.get("story_arc", []), data.get("chapter_ids", []), data.get("user_input", ""))
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    response = await tester.generate_response(incoming_synapse)
    evaluation = evaluate_response(
        response["generated_content"],
        generation_time=1.0,  # Placeholder
        task_type=task_type
    )
    print(json.dumps(evaluation, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
