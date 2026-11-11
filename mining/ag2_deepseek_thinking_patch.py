"""Preserve DeepSeek V4 reasoning_content in ag2 tool-call loops.

ag2 0.9.x _append_oai_message only keeps content/tool_calls/... and drops
reasoning_content. DeepSeek thinking mode requires that field to be replayed
after any tool-call turn (otherwise API may 400).
"""

from __future__ import annotations

_PATCHED = False

_OAI_KEYS = (
    "content",
    "function_call",
    "tool_calls",
    "tool_responses",
    "tool_call_id",
    "name",
    "context",
    "reasoning_content",  # DeepSeek V4 thinking
)


def apply_ag2_reasoning_content_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from autogen.agentchat.conversable_agent import ConversableAgent
    except ImportError:
        print("[ag2-patch] ConversableAgent unavailable; skip reasoning_content patch")
        return False

    def _append_oai_message(self, message, role, conversation_id, is_sending: bool) -> bool:
        message = self._message_to_dict(message)
        oai_message = {
            k: message[k] for k in _OAI_KEYS if k in message and message[k] is not None
        }
        if "content" not in oai_message:
            if "function_call" in oai_message or "tool_calls" in oai_message:
                oai_message["content"] = None
            else:
                return False

        if message.get("role") in ["function", "tool"]:
            oai_message["role"] = message.get("role")
            if "tool_responses" in oai_message:
                for tool_response in oai_message["tool_responses"]:
                    tool_response["content"] = str(tool_response["content"])
        elif "override_role" in message:
            oai_message["role"] = message.get("override_role")
        else:
            oai_message["role"] = role

        if oai_message.get("function_call", False) or oai_message.get("tool_calls", False):
            oai_message["role"] = "assistant"
        elif "name" not in oai_message:
            oai_message["name"] = self.name if is_sending else conversation_id.name

        self._oai_messages[conversation_id].append(oai_message)
        return True

    ConversableAgent._append_oai_message = _append_oai_message
    _PATCHED = True
    print("[ag2-patch] reasoning_content preserved in tool-call history")
    return True
