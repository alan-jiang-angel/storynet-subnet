import json
import urllib.request
import urllib.error

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


# Example usage
if __name__ == "__main__":
    path = "miner_output.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        ## Read a speicific line
        # line_number = 130 # blueprint
        line_number = 133 # characters
        # line_number = 135 # story_arc
        # line_number = 28 # chapters
        
        for i, line in enumerate(f):
            if i == line_number:
                obj = json.loads(line)
                break

    # obj = json.loads(selected)
    
    data = obj.get("data")
    print(data)
    result = send_score_request(data, data.get("task_type"))
    
    print(result)
