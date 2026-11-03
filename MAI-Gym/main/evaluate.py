from pathlib import Path
import json

import requests


from agent import (
    make_observation_message,
    request_model,
    parse_action,
)

def run_episode(
    env,
    task:dict,
    endpoint:str,
    output_dir:str,
    system_prompt:str,
    max_steps:int,
    max_tokens:int,
)->dict:
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)

    messages=[{"role":"system","content":system_prompt}]
    step=0
    stage="reset"

    def save_json(
        name,value
    ):
        path=output_dir/name
        path.write_text(
            json.dumps(value,ensure_ascii=False,indent=2),
            encoding="utf-8",
        )

    save_json('task.json',task)
    save_json('settings.json',{
        "endpoint":endpoint,
        "system_prompt":system_prompt,
        "max_tokens":max_tokens,
        "max_steps":max_steps,
        "temperature":0
    })

    try:
        """OSWorld的标准返回：
        {
    "screenshot": bytes,
    "accessibility_tree": str | None,
    "terminal": str | None,
    "instruction": str
        }"""
        observation=env.reset(task_config=task)
        termination="max_steps"
