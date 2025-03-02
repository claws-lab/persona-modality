import asyncio
import json
import argparse
import logging
import aiofiles
import os
import re
import uuid
import base64
import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import nltk

from litellm import completion, supports_vision

from llm_guard.output_scanners import NoRefusal
from llm_guard.output_scanners.no_refusal import MatchType

nltk.download('punkt', quiet=True)
load_dotenv()



"""
Global Defaults
---------------
These values can be overridden by command-line arguments.
"""

# Generation defaults
DEFAULT_PERSONAS_FILE = 'src/datasets/full-dataset.json'
DEFAULT_QUESTIONS_FILE = 'src/datasets/misc/questions.json'
DEFAULT_SCENARIOS_FILE = 'src/datasets/misc/scenarios.json'
DEFAULT_GENERATION_OUTPUT = f'output/responses-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
DEFAULT_GENERATION_CONCURRENCY = 50

# Refusal evaluation defaults
DEFAULT_REFUSAL_CONCURRENCY = 40
DEFAULT_REFUSAL_THRESHOLD = 0.5
DEFAULT_REFUSAL_MATCH_TYPE = "FULL"
DEFAULT_REFUSAL_OUTPUT = f'output/refusal-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'

# Rubric evaluation defaults
DEFAULT_RUBRIC_CONCURRENCY = 40
RUBRIC_PATHS_DEFAULT = {
    'Linguistic Habits': 'src/rubrics/Linguistic Habits.txt',
    'Persona Consistency': 'src/rubrics/Persona Consistency.txt',
    'Expected Action': 'src/rubrics/Expected Action.txt',
    'Action Justification': 'src/rubrics/Action Justification.txt'
}
SCORE_REGEX = r"Therefore, the final score is\s*(\d+(?:\.\d+)?)"
DEFAULT_RUBRIC_OUTPUT = f'output/rubric-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'

"""
Modality Mapping
----------------
Map numbers to test type strings. You can pass a comma-separated list of numbers.
"""
MODALITY_MAP = {
    "1": "Text",
    "2": "Assisted Image",
    "3": "Image",
    "4": "Descriptive Image"
}
DEFAULT_MODALITIES = list(MODALITY_MAP.values())

# ---------------------------
# Response Generation Functions
# ---------------------------
async def encode_image(image_path):
    async with aiofiles.open(image_path, 'rb') as f:
        data = await f.read()
    return base64.b64encode(data).decode('utf-8')

async def generate_response(prompt, model_name, image=None, system_prompt=None, persona_message=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image:
        encoded = await encode_image(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": persona_message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": persona_message})
    messages.append({"role": "user", "content": prompt})
    max_retries = 10
    backoff_base = 5
    for attempt in range(max_retries):
        try:
            result = await asyncio.to_thread(completion, model=model_name, messages=messages)
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_base * (attempt * 2)
                logging.warning(f"Generation error: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                logging.error(f"Max retries reached during generation. Error: {e}")
                raise e

async def process_persona_generation(persona, model_to_evaluate, questions, scenarios, test_types, semaphore):
    async with semaphore:
        persona_id = persona.get('id')
        persona_name = persona.get('persona')
        description = persona.get('description')
        images = persona.get('images', [])
        persona_detailed = {
            'ID': persona_id,
            'Persona': persona_name,
            'Image Paths': images,
            'Responses': {}
        }
        SYSTEM = (
            "Immerse yourself in the given persona and describe how you would react in the given situation "
            "as if you were genuinely that person. Do not answer as a language model; respond exclusively as if "
            "you were living their life."
        )
        for test_type in test_types:
            if test_type == 'Text':
                persona_message = description
                image_input = None
            elif test_type == 'Image':
                persona_message = "The following image represents your persona."
                image_input = images[0] if images else None
            elif test_type == 'Assisted Image':
                persona_message = description
                image_input = images[0] if images else None
            elif test_type == 'Descriptive Image':
                persona_message = "The following image represents your persona."
                image_input = images[1] if len(images) > 1 else None
            else:
                logging.warning(f"Unknown test type: {test_type}")
                continue
            persona_detailed['Responses'].setdefault(test_type, {})
            # Process Questions
            for category, category_questions in questions.items():
                persona_detailed['Responses'][test_type].setdefault(category, [])
                for question in category_questions:
                    prompt_text = f"{question}"
                    response_text = await generate_response(
                        prompt=prompt_text,
                        image=image_input,
                        system_prompt=SYSTEM,
                        persona_message=persona_message,
                        model_name=model_to_evaluate
                    )
                    persona_detailed['Responses'][test_type][category].append({
                        'Question': question,
                        'Answer': response_text
                    })
            # Process Scenarios
            for category, category_scenarios in scenarios.items():
                persona_detailed['Responses'][test_type].setdefault(category, [])
                for scenario in category_scenarios:
                    prompt_text = f"{scenario}"
                    response_text = await generate_response(
                        prompt=prompt_text,
                        image=image_input,
                        system_prompt=SYSTEM,
                        persona_message=persona_message,
                        model_name=model_to_evaluate
                    )
                    persona_detailed['Responses'][test_type][category].append({
                        'Scenario': scenario,
                        'Answer': response_text
                    })
        return persona_detailed

async def run_generation_pipeline(personas_file, questions_file, scenarios_file, output_file, model_to_evaluate, test_types, concurrency, num_personas=None):
    async with aiofiles.open(personas_file, 'r') as f:
        personas_data = await f.read()
    personas = json.loads(personas_data)
    if num_personas is not None:
        personas = personas[:num_personas]
    async with aiofiles.open(questions_file, 'r') as f:
        questions = json.loads(await f.read())
    async with aiofiles.open(scenarios_file, 'r') as f:
        scenarios = json.loads(await f.read())
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(process_persona_generation(persona, model_to_evaluate, questions, scenarios, test_types, semaphore))
             for persona in personas]
    responses = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Generating responses'):
        responses.append(await task)
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(responses, indent=4))
    logging.info(f"Generation complete. Responses saved to {output_file}.")
    return output_file

# ---------------------------
# Refusal Evaluation Functions
# ---------------------------
refusal_semaphore = asyncio.Semaphore(DEFAULT_REFUSAL_CONCURRENCY)

async def evaluate_refusal(qa, threshold, match_type_str):
    try:
        answer = qa.get('Answer', '')
        question_or_scenario = qa.get('Question') or qa.get('Scenario', '')
        mt = MatchType.FULL if match_type_str.upper() == "FULL" else MatchType.PARTIAL
        scanner = NoRefusal(threshold=threshold, match_type=mt)
        _, is_valid, risk_score = scanner.scan(question_or_scenario, answer)
        evaluation = {
            'Answer': answer,
            'Refusal': not is_valid,
            'Risk Score': risk_score
        }
        if 'Question' in qa:
            evaluation['Question'] = question_or_scenario
        else:
            evaluation['Scenario'] = question_or_scenario
        return evaluation
    except Exception as e:
        logging.error(f"Error in refusal evaluation: {e}")
        return None

async def process_persona_refusal(persona, threshold, match_type_str):
    async with refusal_semaphore:
        persona_id = persona.get('ID')
        persona_name = persona.get('Persona', '')
        responses = persona.get('Responses', {})
        evaluated_persona = {
            'ID': persona_id,
            'Persona': persona_name,
            'Evaluations': {}
        }
        for test_type, categories in responses.items():
            evaluated_persona['Evaluations'][test_type] = {}
            for category, qa_list in categories.items():
                evaluated_persona['Evaluations'][test_type][category] = []
                tasks = [asyncio.create_task(evaluate_refusal(qa, threshold, match_type_str)) for qa in qa_list]
                results = []
                with tqdm(total=len(tasks), desc=f"Refusal eval - {test_type} - {category}", leave=False) as pbar:
                    for fut in asyncio.as_completed(tasks):
                        try:
                            results.append(await fut)
                        except Exception as e:
                            logging.error(f"Error evaluating refusal: {e}")
                        finally:
                            pbar.update(1)
                evaluated_persona['Evaluations'][test_type][category] = results
        return evaluated_persona

async def run_refusal_pipeline(input_file, output_file, threshold, match_type_str):
    logging.info(f"Running refusal pipeline on {input_file}")
    async with aiofiles.open(input_file, 'r') as f:
        data = await f.read()
    personas = json.loads(data)
    tasks = [asyncio.create_task(process_persona_refusal(persona, threshold, match_type_str)) for persona in personas]
    evaluated = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing refusal evaluations"):
        evaluated.append(await task)
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(evaluated, indent=4))
    logging.info(f"Refusal pipeline complete. Output saved to {output_file}.")
    return output_file

# ---------------------------
# Rubric Evaluation Functions
# ---------------------------
def load_rubric_templates(rubric_paths):
    templates = {}
    for rubric, path in rubric_paths.items():
        try:
            with open(path, 'r') as f:
                templates[rubric] = f.read()
        except Exception as e:
            logging.error(f"Error loading template for {rubric} from {path}: {e}")
            templates[rubric] = "{persona}\n{question}\n{response}\n{score_example}"
    return templates

def parse_score_from_response(text):
    try:
        matches = re.findall(SCORE_REGEX, text)
        return float(matches[-1]) if matches else 0
    except Exception as e:
        logging.error(f"Error parsing score: {e}")
        return 0

def prepare_rubric_requests(personas, rubric_templates):
    all_requests = []
    request_info = {}
    for i, persona in enumerate(personas):
        evaluations = persona.get("Evaluations", {})
        for test_type, categories in evaluations.items():
            for category, qa_list in categories.items():
                for j, qa in enumerate(qa_list):
                    if qa.get("Refusal", False):
                        continue  # Skip refusals.
                    if "Question" in qa:
                        rubrics = ["Linguistic Habits", "Persona Consistency", "Expected Action", "Action Justification"]
                        score_example = "Score examples for a typical question..."
                        prompt_field = "Question"
                    else:
                        rubrics = ["Linguistic Habits", "Expected Action", "Action Justification", "Overall Evaluation"]
                        score_example = "Score examples for a typical scenario..."
                        prompt_field = "Scenario"
                    for rubric in rubrics:
                        template = rubric_templates.get(rubric, "{persona}\n{question}\n{response}\n{score_example}")
                        rubric_content = template.format(
                            persona=persona.get("Persona", ""),
                            question=qa.get(prompt_field, ""),
                            response=qa.get("Answer", ""),
                            score_example=score_example
                        )
                        custom_id = f"rubric_{uuid.uuid4().hex}"
                        request_object = {"custom_id": custom_id, "prompt": rubric_content}
                        all_requests.append(request_object)
                        request_info[custom_id] = (i, test_type, category, j, rubric)
    return all_requests, request_info

def update_personas_with_rubric_scores(personas, request_info, results_mapping):
    for custom_id, info in request_info.items():
        persona_index, test_type, category, qa_index, rubric = info
        score = results_mapping.get(custom_id, 0)
        binary = 1 if score >= 3 else 0
        personas[persona_index]["Evaluations"][test_type][category][qa_index][rubric] = {"Score": score, "Binary": binary}
    return personas

async def process_rubric_request(request, semaphore, model_name):
    custom_id = request["custom_id"]
    prompt = request["prompt"]
    async with semaphore:
        try:
            result = await asyncio.to_thread(completion, model=model_name, messages=[{"role": "user", "content": prompt}])
            response_text = result["choices"][0]["message"]["content"]
            score = parse_score_from_response(response_text)
        except Exception as e:
            logging.error(f"Error processing rubric request {custom_id}: {e}")
            score = 0
    return custom_id, score

async def run_rubric_pipeline(input_file, output_file, rubric_templates, concurrency, model_name):
    logging.info(f"Running rubric pipeline on {input_file}")
    async with aiofiles.open(input_file, 'r') as f:
        data = await f.read()
    personas = json.loads(data)
    all_requests, request_info = prepare_rubric_requests(personas, rubric_templates)
    logging.info(f"Prepared {len(all_requests)} rubric evaluation requests.")
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(process_rubric_request(req, semaphore, model_name)) for req in all_requests]
    local_results = {}
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing rubric requests"):
        custom_id, score = await future
        local_results[custom_id] = score
    updated_personas = update_personas_with_rubric_scores(personas, request_info, local_results)
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(updated_personas, indent=4))
    logging.info(f"Rubric pipeline complete. Output saved to {output_file}.")
    return output_file

# ---------------------------
# Main Combined Pipeline
# ---------------------------

def parse_modalities(modality_str):
    """
    Parse a comma-separated list of modality numbers and return the corresponding test types.
    If no valid modalities are provided, return the default list.
    """
    if modality_str:
        chosen = []
        for m in modality_str.split(','):
            m = m.strip()
            if m in MODALITY_MAP:
                chosen.append(MODALITY_MAP[m])
        return chosen if chosen else DEFAULT_MODALITIES
    return DEFAULT_MODALITIES

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Combined Pipeline: Generation -> Refusal -> Rubric Evaluation')
    
    # Generation arguments.
    parser.add_argument('--personas_file', type=str, default=DEFAULT_PERSONAS_FILE, help='Path to personas JSON file.')
    parser.add_argument('--questions_file', type=str, default=DEFAULT_QUESTIONS_FILE, help='Path to questions JSON file.')
    parser.add_argument('--scenarios_file', type=str, default=DEFAULT_SCENARIOS_FILE, help='Path to scenarios JSON file.')
    parser.add_argument('--generation_output', type=str, default=DEFAULT_GENERATION_OUTPUT, help='Output file for generated responses.')
    parser.add_argument('--model_to_evaluate', type=str, required=True, help='LLM model for response generation.')
    parser.add_argument('--generation_concurrency', type=int, default=DEFAULT_GENERATION_CONCURRENCY, help='Concurrency for generation.')
    parser.add_argument('--num_personas', type=int, default=None, help='Number of personas to evaluate (default: all).')
    parser.add_argument('--modalities', type=str, default="1,2,3,4", help='Comma-separated modality numbers (1: Text, 2: Assisted Image, 3: Image, 4: Descriptive Image).')
    
    # Refusal evaluation arguments.
    parser.add_argument('--refusal_threshold', type=float, default=DEFAULT_REFUSAL_THRESHOLD, help='Threshold for refusal detection.')
    parser.add_argument('--refusal_match_type', type=str, default=DEFAULT_REFUSAL_MATCH_TYPE, help='Match type for refusal detection (FULL or PARTIAL).')
    parser.add_argument('--refusal_concurrency', type=int, default=DEFAULT_REFUSAL_CONCURRENCY, help='Concurrency for refusal pipeline.')
    parser.add_argument('--refusal_output', type=str, default=DEFAULT_REFUSAL_OUTPUT, help='Output file for refusal evaluation.')
    
    # Rubric evaluation arguments.
    parser.add_argument('--rubric_concurrency', type=int, default=DEFAULT_RUBRIC_CONCURRENCY, help='Concurrency for rubric evaluation.')
    parser.add_argument('--rubric_output', type=str, default=DEFAULT_RUBRIC_OUTPUT, help='Output file for rubric evaluation.')
    parser.add_argument('--rubric_paths', type=json.loads, default=json.dumps(RUBRIC_PATHS_DEFAULT),
                        help='JSON string mapping rubric names to file paths.')
    parser.add_argument('--evaluator_model', type=str, required=True, help='LLM model to use for rubric evaluation.')
    
    args = parser.parse_args()
    
    # Parse modalities into test types.
    test_types = parse_modalities(args.modalities)
    logging.info(f"Evaluating modalities: {test_types}")
    
    # Step 1: Generation
    generated_file = await run_generation_pipeline(
        personas_file=args.personas_file,
        questions_file=args.questions_file,
        scenarios_file=args.scenarios_file,
        output_file=args.generation_output,
        model_to_evaluate=args.model_to_evaluate,
        test_types=test_types,
        concurrency=args.generation_concurrency,
        num_personas=args.num_personas
    )
    
    # Step 2: Refusal Evaluation
    os.makedirs(os.path.dirname(args.refusal_output) or ".", exist_ok=True)
    await run_refusal_pipeline(
        input_file=generated_file,
        output_file=args.refusal_output,
        threshold=args.refusal_threshold,
        match_type_str=args.refusal_match_type
    )
    
    # Step 3: Rubric Evaluation
    try:
        rubric_paths = json.loads(args.rubric_paths) if isinstance(args.rubric_paths, str) else args.rubric_paths
    except Exception as e:
        logging.error(f"Error parsing rubric_paths: {e}")
        rubric_paths = RUBRIC_PATHS_DEFAULT
    templates = load_rubric_templates(rubric_paths)
    os.makedirs(os.path.dirname(args.rubric_output) or ".", exist_ok=True)
    await run_rubric_pipeline(
        input_file=args.refusal_output,
        output_file=args.rubric_output,
        rubric_templates=templates,
        concurrency=args.rubric_concurrency,
        model_name=args.evaluator_model
    )

if __name__ == "__main__":
    asyncio.run(main())