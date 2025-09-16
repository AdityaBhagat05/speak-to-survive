

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import time
import json
import re
from audio_utils import record_audio, speech_to_text, text_to_speech
from camera_utils import detect_posture_and_confidence
from langchain_ollama import ChatOllama
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    posture_history: list
    start_time: float
    evaluation_done: bool
    pass_meter: int
    turns: int
    last_question: str
    last_user_response: str
    last_transcript: str
    max_turns: int
    meeting_topic: str  # Add this line
llm = ChatOllama(model="mistral:instruct", temperature=0.0)
isPPT=True

def stt_node(state: AgentState) -> AgentState:
    audio_file = record_audio()
    text = speech_to_text(audio_file)
    try:
        os.remove(audio_file)
    except Exception:
        pass
    if not text:
        return {
            "messages": state["messages"],
            "posture_history": state.get("posture_history", []),
            "start_time": state.get("start_time", time.time()),
            "evaluation_done": state.get("evaluation_done", False),
            "pass_meter": state.get("pass_meter", 0),
            "turns": state.get("turns", 0),
            "last_question": state.get("last_question", ""),
            "last_user_response": state.get("last_user_response", ""),
            "last_transcript": state.get("last_transcript", ""),
            "max_turns": state.get("max_turns", 10),
        }
    if text.strip().lower() in ["exit", "quit", "stop"]:
        new_messages = list(state["messages"]) + [HumanMessage(content="exit")]
        return {
            "messages": new_messages,
            "posture_history": state.get("posture_history", []),
            "start_time": state.get("start_time", time.time()),
            "evaluation_done": state.get("evaluation_done", False),
            "pass_meter": state.get("pass_meter", 0),
            "turns": state.get("turns", 0),
            "last_question": state.get("last_question", ""),
            "last_user_response": text,
            "last_transcript": text,
            "max_turns": state.get("max_turns", 10),
        }
    if text.strip() == state.get("last_transcript", ""):
        return {
            "messages": state["messages"],
            "posture_history": state.get("posture_history", []),
            "start_time": state.get("start_time", time.time()),
            "evaluation_done": state.get("evaluation_done", False),
            "pass_meter": state.get("pass_meter", 0),
            "turns": state.get("turns", 0),
            "last_question": state.get("last_question", ""),
            "last_user_response": state.get("last_user_response", ""),
            "last_transcript": text,
            "max_turns": state.get("max_turns", 10),
        }
    human_msg = HumanMessage(content=text)
    new_state = list(state["messages"]) + [human_msg]
    return {
        "messages": new_state,
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": state.get("pass_meter", 0),
        "turns": state.get("turns", 0),
        "last_question": state.get("last_question", ""),
        "last_user_response": text,
        "last_transcript": text,
        "max_turns": state.get("max_turns", 10),
    }


def llm_node(state: AgentState) -> AgentState:
    import re, time

    global llm
    messages = list(state.get("messages", []))
    current_pass_meter = state.get("pass_meter", 0)
    meeting_topic = state.get("meeting_topic", "the presentation")

    system_instruction = (
        "You are an audience member in a public presentation. OUTPUT EXACTLY ONE QUESTION ONLY. "
        "One sentence, <=25 words, end with a question mark, nothing else. "
        "Ask questions about different aspects of the original presentation topic. "
        "Don't just ask follow-ups to the user's last response. "
        "If you cannot think of a relevant question, output exactly: 'Can you give one specific example?'\n"
        f"The presentation topic is: {meeting_topic}\n"
    )

    pass_meter_context = f"\nCurrent user performance score: {current_pass_meter}."
    if current_pass_meter <= -4:
        pass_meter_context += " The user is performing very poorly; be dismissive (still only one question)."
    elif current_pass_meter <= -2:
        pass_meter_context += " The user is performing poorly; be uninterested (still only one question)."
    elif current_pass_meter <= 0:
        pass_meter_context += " The user is neutral; be curious (1 short question)."
    else:
        pass_meter_context += " The user is performing well; be active and respectful (1 short question)."

    system_prompt = system_instruction + pass_meter_context

    history = [m for m in messages if not isinstance(m, SystemMessage)]
    modified_messages = [SystemMessage(content=system_prompt)] + history

    try:
        resp = llm.invoke(messages=modified_messages)
    except TypeError:
        resp = llm.invoke(modified_messages)
    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate)).strip()
    assistant_text = re.sub(r'\s+', ' ', assistant_text).strip()

    def fallback_question(human_text: str) -> str:
        return "Can you give one specific example?"

    q_match = re.search(r'([^?]*\?)', assistant_text)
    first_question = q_match.group(1).strip() if q_match else None

    accept = False
    if first_question:
        word_count = len(first_question.split())
        if word_count <= 25:
            accept = True
            assistant_text = first_question
    else:
        if assistant_text.endswith('?') and len(assistant_text.split()) <= 25:
            accept = True

    if not accept:
        assistant_text = fallback_question(getattr(history[-1], "content", "") if history else "")

    if not assistant_text.endswith('?'):
        assistant_text = assistant_text.rstrip('.') + '?'

    words = assistant_text.split()
    if len(words) > 25:
        assistant_text = ' '.join(words[:25]) + '?'

    ai_msg = AIMessage(content=assistant_text)
    new_messages = messages + [ai_msg]

    return {
        "messages": new_messages,
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": current_pass_meter,
        "turns": state.get("turns", 0),
        "last_question": assistant_text,
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
        "meeting_topic": state.get("meeting_topic", ""),
    }

def tts_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not msgs:
        return state
    last = msgs[-1]
    text = getattr(last, "content", str(last))
    if not text:
        return state
    if text.strip().lower() in ["exit", "quit", "stop", "goodbye!"]:
        return state
    if "Evaluation:" in text:
        text = text.split("Evaluation:")[0].strip()
    try:
        text_to_speech(text)
    except Exception as e:
        print("TTS error:", e)
    return state

def continue_conv(state: AgentState) -> str:
    msgs = state["messages"]
    if msgs and getattr(msgs[-1], "content", "").strip().lower() == "exit":
        return "end"
    if isinstance(msgs[-1], SystemMessage) and "goodbye" in msgs[-1].content.lower():
        return "end"
    max_turns = state.get("max_turns", 10)
    if state.get("turns", 0) >= max_turns:
        print(f"Reached max turns ({max_turns}). Ending.")
        return "end"
    elapsed = time.time() - state.get("start_time", time.time())
    if elapsed >= 300:
        print("⏰ Timer ended: 5 minutes reached, moving to evaluation.")
        return "end"
    return "continue"

def posture_info_node(state: AgentState) -> AgentState:
    if "posture_history" not in state:
        state["posture_history"] = []
    try:
        data = detect_posture_and_confidence()
    except Exception as e:
        data = {"posture": "unknown", "gaze": "unknown", "confidence": "unknown", "arms": "unknown", "head_tilt": None}
        print("Posture detection error:", e)
    state["posture_history"].append(data)
    print(f"[Posture Info] {data}")
    return {
        "messages": state["messages"],
        "posture_history": state["posture_history"],
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": state.get("pass_meter", 0),
        "turns": state.get("turns", 0),
        "last_question": state.get("last_question", ""),
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
    }

def evaluation_node(state: AgentState) -> AgentState:
    msgs = list(state["messages"])
    posture_data = state.get("posture_history", [])
    current_pass_meter = state.get("pass_meter", 0)

    # ✅ Only target the latest HumanMessage
    last_human_msg = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage) and hasattr(m, "content")),
        None
    )
    conversation_text = f"HUMAN: {last_human_msg.content}" if last_human_msg else ""

    evaluation_prompt_text = f"""You are an objective evaluator of public speaking delivery.
Latest response (only human line shown):
{conversation_text}

Scoring rubric (delivery only):
- Provide a numeric score from 0 to 100 (0 worst, 100 best).
- Score is composed of attention (25), clarity (25), grammar (25), persuasiveness (25).
Return EXACTLY one JSON object and nothing else, like:
{{"decision":"PASS" or "FAIL", "score": 0-100, "explanation": "brief (max 200 chars) about delivery"}}
Decision rule: PASS if score >= 60, FAIL otherwise.
"""
    try:
        resp = llm.invoke(messages=[HumanMessage(content=evaluation_prompt_text)])
    except TypeError:
        resp = llm.invoke([HumanMessage(content=evaluation_prompt_text)])
    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate)).strip()
    json_match = re.search(r'(\{.*\})', assistant_text, re.DOTALL)

    decision = "FAIL"
    explanation = ""
    score = 0
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            score = int(parsed.get("score", 0))
            decision = parsed.get("decision", "FAIL").strip().upper()
            explanation = parsed.get("explanation", "")[:200]
        except Exception:
            decision = "FAIL"
            explanation = "Could not parse evaluator JSON."
    else:
        decision = "FAIL"
        explanation = assistant_text.replace("\n", " ")[:200]

    if decision == "PASS":
        new_pass_meter = current_pass_meter + 1
    else:
        new_pass_meter = current_pass_meter - 1

    new_turns = state.get("turns", 0) + 1
    print(f"Evaluation: {decision}. score={score}. {explanation}")

    return {
        "messages": msgs,
        "posture_history": posture_data,
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": new_pass_meter,
        "turns": new_turns,
        "last_question": state.get("last_question", ""),
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
    }


def final_evaluation(state: AgentState) -> AgentState:
    current_pass_meter = state.get("pass_meter", 0)
    print(f"Pass meter final value: {current_pass_meter}")
    if current_pass_meter >= 0:
        print("Passed")
    else:
        print("Failed")
    return {
        "messages": state["messages"],
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": True,
        "pass_meter": current_pass_meter,
        "turns": state.get("turns", 0),
        "last_question": state.get("last_question", ""),
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
    }

graph = StateGraph(AgentState)
graph.add_node("stt", stt_node)
graph.add_node("camera", posture_info_node)
graph.add_node("llm", llm_node)
graph.add_node("tts", tts_node)
graph.add_node("evaluation", evaluation_node)
graph.add_node("final_evaluation", final_evaluation)
graph.add_edge(START, "stt")
graph.add_edge("stt", "camera")
graph.add_edge("camera", "llm")
graph.add_edge("llm", "tts") 
graph.add_edge("tts", "evaluation")
graph.add_conditional_edges(
    "evaluation",
    lambda state: ("continue" if continue_conv(state) == "continue" else "end"),
    {
        "continue": "stt",
        "end": "final_evaluation",
    },
)
graph.add_edge("final_evaluation", END)
app = graph.compile()
if __name__ == "__main__":
    meeting_topic = "is milk chocolate better than dark chocolate?"

    seed: AgentState = {
        "messages": [
            SystemMessage(content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the audience in a public presentation presented by the user. 
You have to ask questions to the user based on their presentation.
Ask only 1 question from a subtopic. DO NOT ask follow up questions to that question.
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Your tone should adapt based on the user's performance (pass_meter value).
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be firm and direct, but curious.
- Push back, challenge their arguments, and make them defend themselves.  
- Keep your responses very short and to the point — no more than 1-2 sentences.
- If the user's performance is poor (negative pass_meter), be increasingly rude and dismissive.
-Use 1 or 2 short sentences only in your question.
""")
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0,
        "turns": 0,
        "last_question": "",
        "last_user_response": "",
        "last_transcript": "",
        "max_turns": 10,
        "meeting_topic": meeting_topic,  # Add this line
    }
    final_state = app.invoke(seed)
    final_pass_meter = final_state.get("pass_meter", 0)
    print(f"\n--- Final Result ---")
    print(f"Pass meter: {final_pass_meter}")
    if final_pass_meter >= 0:
        print("Overall: PASSED")
    else:
        print("Overall: FAILED")
    print("Conversation finished.")


