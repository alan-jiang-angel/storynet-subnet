import json
import urllib.request
import urllib.error

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

load_dotenv()

def send_score_request(req_data: dict, task_type: str) -> dict:
    """
    Send POST request to StoryAI score-miners API using standard library only.
    Only requires output_data as input.
    """

    url = "https://api.storyai.art/score-miners/"

    headers = {
        "User-Agent": "PostmanRuntime/7.39.0",
        "Content-Type": "application/json"
    }

    # Build payload inside function
    payload = {
        "netuid": 92,
        "hotkeys": ["5CSRsopYQnGg3i3CBYoGNm8ogLzDQ7U3kESNqVwHUjHTe8SJ"],
        "uids": [63],
        "task_type": task_type,
        "responses": [
            req_data
        ],
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text)

    except urllib.error.HTTPError as e:
        return {
            "error": "HTTPError",
            "status_code": e.code,
            "message": e.reason,
            "response_text": e.read().decode("utf-8")
        }

    except urllib.error.URLError as e:
        return {
            "error": "URLError",
            "message": str(e.reason)
        }

    except json.JSONDecodeError:
        return {
            "error": "InvalidJSON",
            "message": "Response is not valid JSON"
        }


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

        return result
    
    def test_scoring(self, response_json):
        pass


async def main():
    tester = TestRunner()

    path = "reference/miner_input.jsonl"

    test_lines = [
        (0, "blueprint"),
        (9, "characters"),
        (10, "story_arc"),
        (44, "chapters")
    ]

    for line_number, expected_task in test_lines:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == line_number:
                    obj = json.loads(line)
                    break

        data = obj.get("data")
        task_type = data.get("task_type")
        
        print(f"\n{'='*60}")
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
        print(response)
        
        content = json.loads(response.get("generated_content", "{}"))
        content["_model_info"] = {"mode": "local", "name": "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE", "version": "", "provider": "vllm", "parameters": {"url": "https://openrouter.ai/api"}}
        
        print(json.dumps(content, indent=2))
        
        print("Sending scoring request...")    
        result = send_score_request({ "output_data": content }, task_type)
        print(json.dumps(result, indent=2))
    

if __name__ == "__main__":
    asyncio.run(main())
