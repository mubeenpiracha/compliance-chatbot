# backend/tests/functional/test_graph_conversations.py
import pytest
import asyncio
import json
import glob
from typing import List, Dict

from backend.core.agent_service import get_agent_response
from backend.core.agent.state import AgentState

# --- Test Setup ---

# No need to initialize AI service anymore as it's handled by the agent service

def load_test_cases() -> List[Dict]:
    """Loads all test case JSON files from the test_data directory."""
    test_files = glob.glob("backend/tests/functional/test_data/*.json")
    test_cases = []
    for file_path in test_files:
        with open(file_path, 'r') as f:
            test_case = json.load(f)
            # Add the filename for easier identification of failed tests
            test_case["filename"] = file_path
            test_cases.append(test_case)
    return test_cases

# --- The Test Function ---

@pytest.mark.parametrize("test_case", load_test_cases())
def test_agent_conversation_flow(test_case: Dict):
    """
    Runs a single test case against the agent and asserts the expected outcomes.
    """
    print(f"\n--- Running Test Case: {test_case['name']} ---")

    # 1. Prepare the inputs for the get_ai_response function
    user_message = test_case["user_message"]
    jurisdiction = test_case["jurisdiction"]
    
    # Convert history from test file format to the format expected by the function
    history = []
    for msg in test_case.get("chat_history", []):
        sender = "user" if msg["role"] == "user" else "assistant"
        history.append({"sender": sender, "text": msg["content"]})

    # 2. Invoke the agent
    final_state = invoke_agent_for_test(user_message, history, jurisdiction)
    assert final_state is not None, "Agent did not return a final state"

    # --- DEBUGGING: Print the full final state ---
    print("\n--- Final Agent State ---")
    for key, value in final_state.items():
        # Handle BaseMessage objects for cleaner printing
        if isinstance(value, list) and all(isinstance(item, BaseMessage) for item in value):
            print(f"  {key}: {[f'{type(msg).__name__}({msg.content})' for msg in value]}")
        else:
            print(f"  {key}: {value}")
    print("-------------------------")

    # 3. Assert against the expected values
    expected = test_case["expected_values"]
    
    for key, expected_value in expected.items():
        print(f"  - Checking: {key}")
        actual_value = final_state.get(key)
        
        # Assert that the key exists in the final state before checking its value,
        # unless it's a special key for checking counts or content.
        if "_count" in key or "_contains" in key:
            pass  # Handled by specific checks below
        else:
            assert key in final_state, f"Key '{key}' not found in final state for test '{test_case['name']}'"

        if "_contains" in key:
            # For checks that look for substrings within a list or string
            base_key = key.replace("_contains", "")
            actual_value_for_contains = final_state.get(base_key)
            if isinstance(expected_value, list):
                for item in expected_value:
                    # If an item in the list is another list, treat it as an OR condition
                    if isinstance(item, list):
                        assert any(sub_item.lower() in str(actual_value_for_contains).lower() for sub_item in item), \
                            f"Expected one of {item} to be in '{actual_value_for_contains}' for key '{key}' in test '{test_case['name']}'"
                    else:
                        assert item.lower() in str(actual_value_for_contains).lower(), f"Expected '{item}' not in '{actual_value_for_contains}' for key '{key}' in test '{test_case['name']}'"
            else:
                assert expected_value.lower() in str(actual_value_for_contains).lower(), f"Expected '{expected_value}' not in '{actual_value_for_contains}' for key '{key}' in test '{test_case['name']}'"
        elif "_count" in key:
            # For checks that look for counts of items (e.g., sub_questions)
            key_to_check = key.replace("_count", "")
            # Ensure the list exists before checking its length
            assert key_to_check in final_state and isinstance(final_state[key_to_check], list), \
                f"Expected '{key_to_check}' to be a list in final state for test '{test_case['name']}'"
            actual_count = len(final_state[key_to_check])
            assert actual_count >= expected_value, f"Expected at least {expected_value} items for '{key_to_check}', but got {actual_count} in test '{test_case['name']}'"
        else:
            # For direct equality checks
            assert actual_value == expected_value, f"For key '{key}', expected '{expected_value}' but got '{actual_value}' in test '{test_case['name']}'"

# --- Helper function for testing ---

from backend.core.agent.builder import workflow
from backend.core.agent.state import AgentState

def invoke_agent_for_test(user_message: str, history: list, jurisdiction: str) -> AgentState:
    """
    A test-specific wrapper to invoke the agent and return the full final state.
    """
    # Convert history to the expected format
    formatted_history = []
    for msg in history:
        formatted_history.append({
            'sender': msg['sender'],
            'text': msg['text']
        })

    initial_state: AgentState = {
        "user_query": user_message,
        "jurisdiction": jurisdiction,
        "messages": formatted_history,
        "decision": None,
        "search_results": None,
        "final_response": None,
    }

    # Compile the workflow
    graph = workflow.compile()
    
    # Use a config to add a name to the run for easier debugging in LangSmith
    config = {"configurable": {"thread_id": "test-run"}}
    
    final_state = graph.invoke(initial_state, config=config)
    return final_state