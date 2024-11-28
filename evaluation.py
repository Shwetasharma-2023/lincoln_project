from typing import Optional
import os
import ast
import json
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-4IPmM841wCgFlZf1r7yrT3BlbkFJXGjj1aGburUUslfkOthr"
)


class Reasoning(TypedDict):
    """Structured response combining reasoning and final output."""
    reasoning: Annotated[str, ..., "Provide your reasoning or explanation for why you are making this recommendation or decision."]
    output: Annotated[str, ..., "The final recommendation or answer without any additional explanation."]


class Assessment(TypedDict):
    """Assessment of the generated response, including error status and reasons for any mistakes."""
    test: str
    expectation: str
    result: str  
    reasoning: str
    recommendation: Optional[str] 

class Evaluation_Bot(TypedDict):
    """Structured response combining the final output with a detailed assessment."""
    assessment: Assessment



# Test cases with descriptions and expectations
TEST_CASES = {
    'Conversational Management': {
        'description': 'Assesses the bot\'s ability to lead and progress the conversation smoothly, integrating user responses without abrupt changes.',
        'expectation': 'The bot effectively leads and progresses the conversation, maintaining a natural and logical flow. It integrates user responses appropriately, guides the dialogue without abrupt changes, and ensures the user feels engaged and guided without friction.'
    },
    'User Intent Recognition and Adaptability': {
        'description': 'Evaluates how accurately the bot interprets the user\'s needs and adapts its responses based on user input.',
        'expectation': 'The bot accurately interprets the user’s needs and questions, even when they are ambiguous or not explicitly stated. It adjusts its responses based on the user’s input, adapting to changes in topic or unexpected inputs.'
    },
    'Communication Effectiveness': {
        'description': 'Measures the clarity and conciseness of the bot\'s communication, ensuring appropriate language and tone.',
        'expectation': 'The bot communicates clearly and concisely, using language and tone appropriate for the target audience and consistent with the brand’s voice. It avoids unnecessary jargon or verbosity that could confuse the user.'
    },
    'Accuracy of Information': {
        'description': 'Checks that all information provided by the bot is correct and reliable.',
        'expectation': 'All information provided by the bot is correct, up-to-date, and reliable. The bot avoids sharing incorrect or misleading information.'
    },
    'Compliance with Policies and Regulations': {
        'description': 'Verifies that the bot\'s interactions adhere to company policies and legal guidelines.',
        'expectation': 'All interactions adhere to company policies, legal regulations, and ethical guidelines, avoiding prohibited topics or behaviors.'
    },
    'Avoiding Bias and Stereotypes': {
        'description': 'Ensures the bot\'s responses are free from bias or discriminatory language.',
        'expectation': 'The bot provides responses that are free from bias, stereotypes, or discriminatory language. It treats all users fairly and respectfully.'
    },
    'Challenge Handling and Recovery': {
        'description': 'Evaluates the bot\'s ability to handle errors or misunderstandings gracefully.',
        'expectation': 'When the bot encounters an error, misunderstanding, or user objection, it gracefully recovers by apologizing if necessary and redirecting the conversation appropriately. It addresses user concerns effectively without becoming repetitive or insistent.'
    },
    'Driving User to Outcome': {
        'description': 'Measures how effectively the bot guides the user toward a resolution or next step.',
        'expectation': 'The bot effectively guides the user toward a resolution or next step, helping them achieve their goals within the conversation.'
    },
    'Prompting User for Feedback': {
        'description': 'Assesses the bot\'s ability to proactively seek user feedback or confirmation.',
        'expectation': 'The bot proactively asks for the user’s feedback or confirmation to ensure their needs are met and to clarify any uncertainties.'
    },
    'Depth of Information': {
        'description': 'Evaluates whether the bot provides sufficient detail without overwhelming the user.',
        'expectation': 'The bot’s responses provide sufficient detail to answer the user’s query thoroughly without overwhelming them with unnecessary information.'
    },
    'Sales Ability': {
        'description': 'Examines the bot\'s skill in making persuasive and appropriate product or service recommendations.',
        'expectation': 'The bot demonstrates persuasive skills, making appropriate product or service recommendations that align with the user’s interests and context.'
    },
    'Emotional Intelligence': {
        'description': 'Tests the bot\'s ability to recognize and respond appropriately to the user\'s emotions.',
        'expectation': 'The bot recognizes and appropriately responds to the user’s emotional cues, providing empathy and support when necessary.'
    },
    'Personalization and Memory': {
        'description': 'Assesses how the bot personalizes interactions and remembers user details.',
        'expectation': 'The bot personalizes interactions based on user data or previous conversations, remembering details and referencing them appropriately to enhance the user experience.'
    },
    'Problem-Solving Skills': {
        'description': 'Evaluates the bot\'s effectiveness in assisting with resolving issues or complex questions.',
        'expectation': 'The bot effectively assists the user in resolving issues or answering complex questions through logical reasoning and helpful suggestions.'
    },
    'Closing Conversations Appropriately': {
        'description': 'Checks whether the bot concludes conversations politely and appropriately.',
        'expectation': 'The bot recognizes when a conversation is coming to an end and provides a polite and appropriate closing, ensuring the user feels satisfied with the interaction.'
    }
}

def generate_system_prompt(test_case, expectation, query, agents_prompt, bot_response):
    return f"""
    Please evaluate the chatbot's performance based on the selected test case.

    **Test Case**: {test_case}
    - **Expectation**: {expectation}

    Focus **strictly** on evaluating the bot's performance according to this specific test case. Do not evaluate other criteria unless they are directly relevant to this test.

    Your response should include:
    1. **Result**: Did the bot pass or fail the test based on {test_case}'s criteria?
    2. **Reasoning**: Provide detailed reasoning for your assessment, but limit your explanation strictly to the test case at hand.
    3. **Recommendation**: Include recommendations **only if the bot failed**. Otherwise, omit this section or set it to `null`.

    Here is the format of your response:
    {{
      "assessment": {{
        "test": "{test_case}",
        "expectation": "{expectation}",
        "result": "<Pass or Fail>",
        "reasoning": "<Explanation>",
        "recommendation": "<Suggestions, if applicable>"
      }}
    }}

    ### agent_prompt:
    {agents_prompt}

    ### Bot's Response:
    {bot_response}
    """


def analyze_conversation(conversation_logs):
    # Prepare a summary of conversation logs for the LLM
    conversation_summary = "\n".join([
        f"Query: {log['query']}\nBot Response: {log['bot_response']}"
        for log in conversation_logs
    ])
    
    # Create a prompt to send to the LLM for test selection
    analysis_prompt = f"""
    Based on the following conversation logs, decide which test cases from the list should be applied. Provide a list of relevant test cases and explanations for their selection.
    Output should be a list of test case names, each name appearing only once, without additional explanation or details.

    Conversation Logs:
    {conversation_summary}

    List of Available Test Cases:
    {', '.join(TEST_CASES.keys())}
    
    Output:
    [Test Case Name 1, Test Case Name 2]
    Example Output: ["Conversational Management", ""]

    """
    
    # Use the LLM to analyze and select relevant tests
    analysis_result = model.invoke(analysis_prompt)
    list_testcase= analysis_result.content
    list_testcase = ast.literal_eval(list_testcase)

    
    return list_testcase


model = ChatOpenAI(model="gpt-4o", temperature= 0.0)
structured_evaluation = model.with_structured_output(Evaluation_Bot)

def main():
    conversation_logs = json.loads(open("conversation_logs.json").read())
    evaluate_logs(conversation_logs)

def evaluate_logs(conversation_logs):
    selected_tests = analyze_conversation(conversation_logs)

    for test in selected_tests:
        if test in TEST_CASES:
            fixed_expectation = TEST_CASES[test]['expectation']
            
            print("Selected Test case----", test)
            
            for log in conversation_logs:
                agents_prompt = log.get("agents_prompt")
                bot_response = log.get("bot_response")
                
                system_prompt_temp = generate_system_prompt(test, fixed_expectation, log["query"], agents_prompt, bot_response)
                evaluation = structured_evaluation.invoke(system_prompt_temp)

                evaluation["assessment"]["test"] = test
                evaluation["assessment"]["expectation"] = fixed_expectation
                json_output = json.dumps(evaluation, indent=2)
                
                print("Query---------", log["query"])
                print(f"Bot Response----------", bot_response)
                print("Evaluation Bot-----------------", json_output)
                
                # Optionally, you can add a separator between logs for clarity
                print("-" * 40)

if __name__ == '__main__':
    main()
