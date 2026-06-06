# Stub: upstream conversion/mistral.py not picked into fork.
# _set_vocab_mistral / mistral-format paths in conversion/base.py reference
# MistralModel and MistralVocab; this stub satisfies the type-checker without
# enabling Mistral conversion. Calls into these classes raise at runtime.
from typing import Any


class MistralModel:
    @staticmethod
    def get_community_chat_template(vocab: Any, template_dir: Any, is_mistral_format: Any) -> str:
        raise NotImplementedError(
            'Mistral conversion not available in this fork. Install upstream conversion/mistral.py if needed.'
        )


class MistralVocab:
    pass
